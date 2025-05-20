"""Modified from https://github.com/facebookresearch/vggt/blob/main/demo_gradio.py"""

import glob
import json
import os
import shutil
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from evo.core.geometry import umeyama_alignment
from polyform.core.capture_folder import CaptureFolder
from tqdm import tqdm

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from will_utils import (
    voxel_downsample_with_colors,
    process_frame,
    get_point_cloud_for_frame,
)

import rerun as rr


device = "cuda" if torch.cuda.is_available() else "cpu"
if not torch.cuda.is_available():
    raise ValueError("CUDA is not available. Please run on a machine with a GPU.")

print("Initializing and loading VGGT model...")
model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
model.eval().to(device)
print(f"VGGT model loaded successfully to {device}")


def vggt_inference(image_paths: List[str]) -> dict:
    # Load and preprocess images
    torch.cuda.empty_cache()
    img_tensor = load_and_preprocess_images(sorted(image_paths)).to(device)
    print(f"Loaded images to tensor of shape {img_tensor.shape}")

    # Forward pass
    print("Running inference...")
    dtype = (
        torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    )
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        predictions = model(img_tensor)
    forward_dur = time.perf_counter() - start_time
    print(f"Inference completed in {forward_dur:.2f}s")

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], img_tensor.shape[-2:]
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # NOTE: this is HELLA slow so avoid
    # Convert tensors to numpy
    # for key in predictions:
    #     if isinstance(predictions[key], torch.Tensor):
    #         predictions[key] = predictions[key].cpu().numpy().squeeze(0).tolist()

    # Generate world points from depth map
    print("Computing world points from depth map...")
    depth_map = predictions["depth"].cpu().numpy()[0]
    world_points = unproject_depth_map_to_point_map(
        depth_map,
        predictions["extrinsic"].cpu().numpy()[0],
        predictions["intrinsic"].cpu().numpy()[0],
    )
    predictions["world_points_from_depth"] = world_points
    return predictions


def get_point_cloud(
    predictions: dict, conf_percentile: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Extract and reshape colors
    images = predictions["images"][0]
    if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
        colors_rgb = images.permute(0, 2, 3, 1)  # NHWC format
    else:
        colors_rgb = images
    colors_rgb = colors_rgb.reshape(-1, 3) * 255
    colors_rgb = colors_rgb.to(torch.uint8)

    # Pointmap branch
    world_points = predictions["world_points"]
    world_points_conf = predictions["world_points_conf"]

    # Depthmap and camera branch
    world_points = torch.tensor(predictions["world_points_from_depth"], device=device)
    world_points_conf = predictions["depth_conf"]

    points = world_points.reshape(-1, 3)
    points_conf = world_points_conf.reshape(-1)

    conf_threshold = torch.quantile(points_conf, conf_percentile / 100.0)
    conf_mask = (points_conf >= conf_threshold) & (points_conf > 1e-5)
    print(
        "Number of points above confidence threshold:",
        conf_mask.sum(),
        "/",
        len(points_conf),
    )

    good_points = points[conf_mask]
    good_rgbs = colors_rgb[conf_mask]
    assert good_points.shape == good_rgbs.shape
    return good_points, good_rgbs


def apply_similarity_transform_to_poses_vec(
    poses: np.ndarray, R: np.ndarray, t: np.ndarray, s: float
) -> np.ndarray:
    # Extract rotations and translations
    R_pred = poses[:, :3, :3]  # (N, 3, 3)
    t_pred = poses[:, :3, 3]  # (N, 3)

    # Apply rotation: R @ R_pred
    R_new = R @ R_pred  # (3, 3) @ (N, 3, 3) = (N, 3, 3)

    # Apply scaling and translation to translation vectors
    t_new = s * (R @ t_pred.T).T + t  # (N, 3)

    # Assemble new poses
    aligned = np.zeros_like(poses)
    aligned[:, :3, :3] = R_new
    aligned[:, :3, 3] = t_new
    aligned[:, 3, 3] = 1.0
    return aligned


def get_polycam_pcd(frames):
    pts = []
    rgbs = []
    for frame in tqdm(frames):
        new_points, new_colors = get_point_cloud_for_frame(frame)
        pts.append(new_points)
        rgbs.append(new_colors)
    pts = np.concatenate(pts, axis=0)
    rgbs = np.concatenate(rgbs, axis=0)
    return pts, rgbs


def demo():
    rr.init("vggt", spawn=True)
    polycam_dataset = "/home/wilshen/datasets/feijoa/baymax_v1"
    image_dir = os.path.join(polycam_dataset, "keyframes/images")
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))

    folder = CaptureFolder(polycam_dataset)
    keyframes = folder.get_keyframes(rotate=True)
    frames = [process_frame(kf) for kf in keyframes]
    assert image_paths == [f["rgb_path"] for f in frames]

    predictions = vggt_inference(image_paths)

    points, rgbs = get_point_cloud(predictions, conf_percentile=70.0)
    # rr.log("vggt_pcd", rr.Points3D(positions=points.cpu(), colors=rgbs.cpu()))

    # Get predicted extrinsics and align with polycam extrinsics
    vggt_extrinsics = torch.eye(4)[None].repeat(len(image_paths), 1, 1)
    vggt_extrinsics[:, :3, :4] = predictions["extrinsic"][0]
    vggt_c2ws = torch.linalg.inv(vggt_extrinsics)

    polycam_c2ws = np.array([f["c2w"] for f in frames])
    polycam_c2ws = torch.tensor(polycam_c2ws, device=device)

    vggt_xyz = vggt_c2ws[:, :3, 3]
    polycam_xyz = polycam_c2ws[:, :3, 3]

    R, t, s = umeyama_alignment(
        x=vggt_xyz.T.cpu().numpy(),
        y=polycam_xyz.T.cpu().numpy(),
        with_scale=True,
    )

    # Align the VGGT-predicted poses to the Polycam poses
    vggt_aligned_c2ws = apply_similarity_transform_to_poses_vec(
        poses=vggt_c2ws.cpu().numpy(),
        R=R,
        t=t,
        s=s,
    )
    assert vggt_aligned_c2ws.shape == polycam_c2ws.shape

    # Transform the point cloud
    points_np = points.cpu().numpy()
    points_transformed = s * (points_np @ R.T) + t
    points_transformed = torch.tensor(points_transformed, device=points.device)

    # Downsample point cloud
    voxel_size = 0.02
    points_d, rgbs_d = voxel_downsample_with_colors(
        points_transformed, rgbs, voxel_size=voxel_size
    )
    rr.log(
        "vggt_pcd",
        rr.Points3D(
            positions=points_d.cpu(), colors=rgbs_d.cpu(), radii=voxel_size / 2
        ),
    )

    polycam_pts, polycam_rgbs = get_polycam_pcd(frames)
    polycam_pts = torch.tensor(polycam_pts, device=device)
    polycam_rgbs = torch.tensor(polycam_rgbs, device=device)
    polycam_pts_d, polycam_rgbs_d = voxel_downsample_with_colors(
        polycam_pts, polycam_rgbs, voxel_size=voxel_size
    )
    rr.log(
        "polycam_pcd",
        rr.Points3D(
            positions=polycam_pts_d.cpu(),
            colors=polycam_rgbs_d.cpu(),
            radii=voxel_size / 2,
        ),
    )

    for idx, (vggt_c2w, polycam_c2w) in enumerate(
        zip(vggt_aligned_c2ws, polycam_c2ws.cpu())
    ):
        # rr.set_time(timeline="step", sequence=idx)
        rr.log(
            f"cam/vggt/{idx:03d}",
            rr.Transform3D(translation=vggt_c2w[:3, 3], mat3x3=vggt_c2w[:3, :3]),
        )
        rr.log(f"pred_cam/{idx:03d}", [rr.components.AxisLength(0.1)])

        rr.log(
            f"cam/polycam/{idx:03d}",
            rr.Transform3D(translation=polycam_c2w[:3, 3], mat3x3=polycam_c2w[:3, :3]),
        )
        rr.log(f"cam/polycam/{idx:03d}", [rr.components.AxisLength(0.1)])

    # Hack around with intrinsics to apply scale and make sure point cloud is still aligned
    depth_map = predictions["depth"].cpu().numpy()[0].copy()
    depth_map *= s
    vggt_new_w2cs = np.linalg.inv(vggt_aligned_c2ws)
    scaled_points = unproject_depth_map_to_point_map(
        depth_map,
        vggt_new_w2cs,
        predictions["intrinsic"].cpu().numpy()[0],
    )
    images = predictions["images"][0]
    if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
        colors_rgb = images.permute(0, 2, 3, 1)  # NHWC format
    else:
        colors_rgb = images
    colors_rgb = colors_rgb.reshape(-1, 3) * 255
    colors_rgb = colors_rgb.to(torch.uint8)
    scaled_rgbs = colors_rgb

    scaled_points = scaled_points.reshape(-1, 3)
    scaled_points = torch.tensor(scaled_points, device=device)

    conf = predictions["depth_conf"]
    conf = conf.reshape(-1)
    conf_threshold = torch.quantile(conf, 0.0)
    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)

    scaled_points = scaled_points[conf_mask]
    scaled_rgbs = scaled_rgbs[conf_mask]

    down_pts, down_rgbs = voxel_downsample_with_colors(
        scaled_points, scaled_rgbs, voxel_size=voxel_size
    )
    rr.log(
        "scaled_pcd",
        rr.Points3D(positions=down_pts.cpu(), colors=down_rgbs.cpu(), radii=voxel_size / 2),
    )

    out_dir = Path("/tmp/vggt_out")

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    depth_dir = out_dir / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)

    transforms = {"frames": []}
    scaled_depth_maps = depth_map
    intrinsics = predictions["intrinsic"].cpu().numpy()[0]

    rgbs = predictions["images"][0].cpu().numpy()

    for src_path, rgb, depth, c2w, intrinsic in zip(image_paths, rgbs, scaled_depth_maps, vggt_aligned_c2ws, intrinsics):
        dst_path = images_dir / Path(src_path).name
        # Explicit copy
        # shutil.copy(src_path, dst_path)
        rgb = rgb.transpose(1, 2, 0)
        rgb_255 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        # it's a RGB not BGR
        bgr = cv2.cvtColor(rgb_255, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(dst_path), bgr)

        depth_path = depth_dir / Path(src_path).name
        # replace extension with png
        depth_path = depth_path.with_suffix(".png")
        depth_rel_path = depth_path.relative_to(out_dir)

        # Multiply by 1000 to convert to meters then uint16
        depth_map_uint16 = (depth * 1000.0).astype(np.uint16)
        # Save depth map as uint16 png
        cv2.imwrite(str(depth_path), depth_map_uint16)

        dst_rel_path = dst_path.relative_to(out_dir)
        frame = {
            "file_path": str(dst_rel_path),
            "depth_file_path": str(depth_rel_path),
            "transform_matrix": c2w.tolist(),
            "intrinsic_matrix": intrinsic.tolist(),
        }
        transforms["frames"].append(frame)

    conf_raw = predictions["depth_conf"][0].cpu().numpy()
    # Convert to float16
    # conf_raw = conf_raw.astype(np.float16)

    # Save as np array
    conf_raw_path = out_dir / "depth_conf.npy"
    np.save(str(conf_raw_path), conf_raw)

    transforms["depth_confidence_path"] = str(conf_raw_path)

    with open(out_dir / "transforms.json", "w") as f:
        json.dump(transforms, f, indent=2)



    print()

if __name__ == "__main__":
    demo()
