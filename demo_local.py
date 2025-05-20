"""Modified from https://github.com/facebookresearch/vggt/blob/main/demo_gradio.py"""

import glob
import os
import time

import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

device = "cuda" if torch.cuda.is_available() else "cpu"
if not torch.cuda.is_available():
    raise ValueError("CUDA is not available. Please run on a machine with a GPU.")


print("Initializing and loading VGGT model...")
model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
model.eval().to(device)
print(f"VGGT model loaded successfully to {device}")

image_dir = "/home/wilshen/datasets/feijoa/teddy_v1/keyframes/images"  # replace with your actual path
image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))

# Load and preprocess images
torch.cuda.empty_cache()
img_tensor = load_and_preprocess_images(sorted(image_paths)).to(device)
print(f"Loaded images to tensor of shape {img_tensor.shape}")

# Forward pass
print("Running inference...")
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
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

# NOTE: this is HELLA slow
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

conf_thresh = 30.0

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

points = world_points.reshape(-1, 3)
points_conf = world_points_conf.reshape(-1)

# conf_threshold = np.percentile(points_conf, conf_thresh)
# use torch equivalent
conf_threshold = torch.quantile(points_conf, conf_thresh / 100.0)

conf_mask = (points_conf >= conf_threshold) & (points_conf > 1e-5)
print("Number of points above confidence threshold:", conf_mask.sum())

good_points = points[conf_mask]
good_rgbs = colors_rgb[conf_mask]
assert good_points.shape == good_rgbs.shape

import rerun as rr

rr.init("vggt", spawn=True)
rr.log("pcd", rr.Points3D(positions=good_points.cpu(), colors=good_rgbs.cpu()))

for image in predictions["images"][0]:
    chw = image
    hwc = image.permute(2, 1, 0)
    rr.log("image", rr.Image(hwc.cpu()))
# Depthmap and camera branch


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


good_point_down, good_color_down = voxel_downsample_with_colors(
    good_points, good_rgbs, voxel_size=0.02
)

rr.log(
    "pcd_downsampled",
    rr.Points3D(positions=good_point_down.cpu(), colors=good_color_down.cpu()),
)
