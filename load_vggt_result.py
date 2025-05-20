from pathlib import Path
import json
import cv2

from vggt.utils.geometry import unproject_depth_map_to_point_map
import numpy as np
import rerun as rr


def downsample_point_cloud(points, colors, voxel_size: float):
    """Voxel downsample a point cloud.

    Reference: https://github.com/isl-org/Open3D/blob/d7a2cf608a5e206d8ebc3b78d947c219cd4da8fb/cpp/open3d/geometry/PointCloud.cpp#L354
    """
    assert voxel_size > 0, "voxel_size must be positive"
    voxel_min_bound = points.min(0) - voxel_size * 0.5

    ref_coords = (points - voxel_min_bound) / voxel_size
    voxel_idxs = ref_coords.astype(int)
    voxel_idxs, inverse, counts = np.unique(
        voxel_idxs, axis=0, return_inverse=True, return_counts=True
    )

    voxels = np.zeros((voxel_idxs.shape[0], 3))
    np.add.at(voxels, inverse, points)
    voxels /= counts.reshape(-1, 1)

    voxel_colors = np.zeros((voxel_idxs.shape[0], 3))
    np.add.at(voxel_colors, inverse, colors)
    voxel_colors /= counts.reshape(-1, 1)
    voxel_colors = voxel_colors.astype(np.uint8)
    assert voxel_colors.max() <= 255

    return voxels, voxel_colors


def load(conf_percent: float = 50.0, voxel_size: float = 0.02):
    out_dir = Path("/home/wilshen/workspace/vggt/server/results/2025-05-20_15-53-32/outputs")

    transforms_path = out_dir / "transforms.json"
    with open(transforms_path, "r") as f:
        transforms = json.load(f)

    frames = []
    for frame_dict in transforms["frames"]:
        rgb_path = out_dir / frame_dict["file_path"]
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR_RGB)

        depth_path = out_dir / frame_dict["depth_file_path"]
        depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)

        c2w = frame_dict["transform_matrix"]
        intrinsic = frame_dict["intrinsic_matrix"]

        frames.append(
            {
                "rgb": rgb,
                "depth": depth,
                "intrinsic": intrinsic,
                "c2w": c2w,
            }
        )

    # Load confidence
    confidence_path = out_dir / transforms["depth_confidence_path"]
    confidence = np.load(confidence_path)

    # Convert to point cloud
    rgbs = np.array([frame["rgb"] for frame in frames])
    depths = np.array([frame["depth"] for frame in frames])
    assert rgbs.shape[:-1] == depths.shape
    c2ws = np.array([frame["c2w"] for frame in frames])
    w2cs = np.linalg.inv(c2ws)
    intrinsics = np.array([frame["intrinsic"] for frame in frames])

    point_map = unproject_depth_map_to_point_map(
        depth_map=depths[..., None] / 1000,
        extrinsics_cam=w2cs[..., :3, :4],
        intrinsics_cam=intrinsics,
    )
    assert point_map.shape == rgbs.shape

    # Compute confidence threshold
    conf_flat = confidence.reshape(-1)
    conf_threshold = np.percentile(conf_flat, conf_percent)

    rr.init("load", spawn=True)
    all_points = None
    all_colors = None

    frame_count = 0
    for rgb, depth, points, conf, c2w, K in zip(
        rgbs, depths, point_map, confidence, c2ws, intrinsics
    ):
        rr.set_time("frame", sequence=frame_count)

        rr.log("cam/rgb", rr.Image(rgb))
        rr.log("cam/depth", rr.DepthImage(depth, meter=1000.0))
        rr.log("cam", rr.Transform3D(translation=c2w[:3, 3], mat3x3=c2w[:3, :3]))

        mask = conf > conf_threshold
        points_valid = points[mask].reshape(-1, 3)
        rgb_valid = rgb[mask].reshape(-1, 3)
        if all_points is None:
            all_points = points_valid
            all_colors = rgb_valid
        else:
            all_points = np.vstack((all_points, points_valid))
            all_colors = np.vstack((all_colors, rgb_valid))
        all_points, all_colors = downsample_point_cloud(
            all_points, all_colors, voxel_size
        )
        rr.log(
            "pcd",
            rr.Points3D(positions=all_points, colors=all_colors, radii=voxel_size / 2),
        )

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        height, width = rgb.shape[:2]
        rr.log(
            "cam",
            rr.Pinhole(
                focal_length=[fx, fy],
                principal_point=[cx, cy],
                width=width,
                height=height,
            ),
        )
        frame_count += 1


if __name__ == "__main__":
    load(conf_percent=70, voxel_size=0.01)
