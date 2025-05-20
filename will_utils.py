import glob
import os
import time
from typing import Any, Tuple

import cv2
import numpy as np
import torch
from evo.core.geometry import umeyama_alignment
from polyform.core.capture_folder import Keyframe, CaptureFolder

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def convert_opengl_to_opencv(c2w: np.ndarray):
    """Convert from OpenGL to OpenCV coordinate convention."""
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    c2w = np.matmul(c2w, flip_yz)
    return c2w


def process_frame(frame: Keyframe) -> dict[str, Any]:
    """Process polyform keyframe"""
    corrected = frame.is_optimized()
    rgb_path = frame.corrected_image_path if corrected else frame.image_path

    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    depth_path = frame.depth_path
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)

    confidence_path = frame.confidence_path
    confidence = cv2.imread(confidence_path)
    confidence_processed = np.full_like(confidence[..., 0], 3, dtype=np.uint8)
    confidence_processed[confidence[..., 0] == 0] = 0
    confidence_processed[confidence[..., 0] == 54] = 1
    confidence_processed[confidence[..., 0] == 255] = 2
    # Check 3 does not exist anymore
    assert (confidence_processed == 3).sum() == 0

    intrinsic = {
        "cx": frame.camera.cx,
        "cy": frame.camera.cy,
        "fx": frame.camera.fx,
        "fy": frame.camera.fy,
        "width": frame.camera.width,
        "height": frame.camera.height,
    }

    c2w = convert_opengl_to_opencv(frame.camera.transform)
    return {
        "rgb": rgb,
        "depth": depth,
        "confidence": confidence_processed,
        "intrinsic": intrinsic,
        "c2w": c2w,
        "rgb_path": rgb_path,
        "depth_path": depth_path,
    }


def get_point_cloud(
    rgb,
    depth,
    fx,
    fy,
    cx,
    cy,
    c2w=None,
    mask=None,
    depth_scale: float = 1000.0,
    depth_trunc: float = 5.0,
):
    """Convert RGB-D image with given camera intrinsics to a point cloud."""
    # We only consider points within the depth truncation
    z = depth / depth_scale
    if mask is not None:
        combined_mask = np.logical_and(z <= depth_trunc, mask)
    else:
        combined_mask = z <= depth_trunc
    z = z[combined_mask].reshape(-1)

    # Get pixel coordinates
    height, width = rgb.shape[:2]
    vu = np.indices((height, width))
    vu = vu[:, combined_mask].reshape(2, -1)
    assert z.shape[0] == vu.shape[1]

    # Compute x, y in camera coordinates
    x = (vu[1] - cx) * z / fx
    y = (vu[0] - cy) * z / fy
    points = np.vstack((x, y, z)).T

    # Transform to world coordinates
    if c2w is not None:
        points = c2w @ np.hstack((points, np.ones((points.shape[0], 1)))).T
        points = points[:3].T

    # Get colors
    colors = rgb[combined_mask].reshape(-1, 3)
    return points, colors


def get_point_cloud_for_frame(frame: dict) -> Tuple[np.ndarray, np.ndarray]:
    intrinsic = frame["intrinsic"]
    width, height = intrinsic["width"], intrinsic["height"]
    cx, cy = intrinsic["cx"], intrinsic["cy"]
    fx, fy = intrinsic["fx"], intrinsic["fy"]
    c2w = frame["c2w"]
    depth = cv2.resize(frame["depth"], (width, height), interpolation=cv2.INTER_NEAREST)

    confidence = frame["confidence"]
    confidence = cv2.resize(
        confidence, (width, height), interpolation=cv2.INTER_NEAREST
    )
    mask = confidence == 2  # high confidence depth only
    new_points, new_colors = get_point_cloud(
        frame["rgb"], depth, fx=fx, fy=fy, cx=cx, cy=cy, c2w=c2w, mask=mask
    )
    return new_points, new_colors


def voxel_downsample_with_colors(
    points: torch.Tensor, colors: torch.Tensor, voxel_size: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Downsample a point cloud using a voxel grid. Averages both point positions and colors.

    Args:
        points: (N, 3) float tensor of point coordinates.
        colors: (N, 3) uint8 or float tensor of RGB values in [0, 255] or [0, 1].
        voxel_size: float voxel resolution.

    Returns:
        downsampled_points: (M, 3) float tensor of averaged point positions.
        downsampled_colors: (M, 3) uint8 tensor of averaged colors.
    """
    assert points.shape[0] == colors.shape[0]
    assert voxel_size > 0

    if colors.dtype != torch.float32:
        colors = colors.float()

    voxel_min_bound = points.amin(dim=0) - voxel_size * 0.5
    ref_coords = (points - voxel_min_bound) / voxel_size
    voxel_indices = torch.floor(ref_coords).long()

    unique_indices, inverse_indices = torch.unique(
        voxel_indices, dim=0, return_inverse=True
    )

    num_voxels = unique_indices.shape[0]
    point_sum = torch.zeros((num_voxels, 3), dtype=points.dtype, device=points.device)
    color_sum = torch.zeros((num_voxels, 3), dtype=colors.dtype, device=colors.device)
    counts = torch.zeros(num_voxels, dtype=torch.int64, device=points.device)

    point_sum.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, 3), points)
    color_sum.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, 3), colors)
    counts.scatter_add_(0, inverse_indices, torch.ones_like(inverse_indices))

    downsampled_points = point_sum / counts.unsqueeze(1)
    downsampled_colors = (color_sum / counts.unsqueeze(1)).clamp(0, 255).to(torch.uint8)

    return downsampled_points, downsampled_colors
