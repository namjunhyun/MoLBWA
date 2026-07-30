#!/usr/bin/env python3
"""Fuse D455 RGB-D frames with ORB-SLAM3 poses into a colored TSDF map."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import open3d as o3d
from PIL import Image
import rerun as rr
import rerun.blueprint as rrb


TRACKING_OK = {2, 5}
FX = 392.6767578125
FY = 392.6767578125
CX = 321.53253173828125
CY = 242.13388061523438
WIDTH = 640
HEIGHT = 480


def rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    relative = a.T @ b
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


def read_last_line(path: Path) -> str:
    with path.open("rb") as stream:
        stream.seek(0, 2)
        end = stream.tell()
        if end == 0:
            return ""
        position = end - 1
        while position > 0:
            stream.seek(position)
            if stream.read(1) == b"\n" and position < end - 1:
                break
            position -= 1
        stream.seek(position + (1 if position > 0 else 0))
        return stream.readline().decode(errors="replace").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-file", default="/tmp/d455_slam_pose.csv")
    parser.add_argument("--voxel-size", type=float, default=0.025)
    parser.add_argument("--depth-trunc", type=float, default=5.0)
    parser.add_argument("--extract-every", type=int, default=15)
    parser.add_argument("--output", default=str(Path.home() / "d455_tsdf_map.ply"))
    args = parser.parse_args()

    blueprint = rrb.Horizontal(
        rrb.Spatial3DView(name="World", origin="world", contents=["world/**"]),
        rrb.Spatial2DView(name="Camera", origin="world/camera", contents=["world/camera/color"]),
        column_shares=[2, 1],
    )
    rr.init("d455_rgbd_tsdf_map", spawn=False, default_blueprint=blueprint)
    rr.spawn(memory_limit="3GiB", server_memory_limit="512MiB", executable_path=str(Path(sys.executable).with_name("rerun")), default_blueprint=blueprint)
    rr.log("world", rr.ViewCoordinates.RFU, static=True)
    rr.log("world/camera", rr.Pinhole(focal_length=[FX, FY], principal_point=[CX, CY], resolution=[WIDTH, HEIGHT], image_plane_distance=0.18), static=True)
    rr.log("world/camera", rr.ViewCoordinates.RDF, static=True)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        WIDTH, HEIGHT, FX, FY, CX, CY
    )
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_size,
        sdf_trunc=args.voxel_size * 4.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    pose_path = Path(args.pose_file)
    depth_path = Path("/tmp/d455_depth.png")
    color_path = Path("/tmp/d455_color.png")
    output_path = Path(args.output)
    last_pose_line = ""
    last_image_mtime = 0
    last_live_mtime = 0
    last_twc: np.ndarray | None = None
    keyframes = 0
    frame_index = 0

    if output_path.exists():
        saved_cloud = o3d.io.read_point_cloud(str(output_path))
        saved_points = np.asarray(saved_cloud.points, dtype=np.float32)
        saved_colors = np.clip(np.asarray(saved_cloud.colors) * 255.0, 0, 255).astype(np.uint8)
        if len(saved_points):
            rr.log("world/saved_tsdf_map", rr.Points3D(saved_points, colors=saved_colors, radii=args.voxel_size * 0.45), static=True)
            print(f"Loaded saved map with {len(saved_points)} points", flush=True)

    print("Waiting for RGB-D frames and ORB-SLAM3 poses", flush=True)
    try:
        while True:
            if not (pose_path.exists() and depth_path.exists() and color_path.exists()):
                time.sleep(0.05)
                continue

            line = read_last_line(pose_path)
            if not line or line == last_pose_line:
                time.sleep(0.02)
                continue
            parts = line.split(",")
            if len(parts) != 18:
                time.sleep(0.02)
                continue
            last_pose_line = line
            state = int(parts[1])
            tcw = np.asarray(parts[2:], dtype=np.float64).reshape(4, 4)
            twc = np.linalg.inv(tcw)
            frame_index += 1

            rr.set_time("frame", sequence=frame_index)
            rr.log(
                "world/camera",
                rr.Transform3D(translation=twc[:3, 3], mat3x3=twc[:3, :3]),
            )

            image_mtime = min(
                depth_path.stat().st_mtime_ns, color_path.stat().st_mtime_ns
            )
            if image_mtime > last_live_mtime and frame_index % 3 == 0:
                live_depth = np.asarray(Image.open(depth_path), dtype=np.uint16)
                live_color = np.asarray(Image.open(color_path).convert("RGB"), dtype=np.uint8)
                rr.log("world/camera/color", rr.Image(live_color, color_model="RGB"))
                rr.log("world/camera/depth", rr.DepthImage(live_depth, meter=1000.0))
                last_live_mtime = image_mtime
            if state not in TRACKING_OK or image_mtime <= last_image_mtime:
                continue

            translation = (
                np.inf
                if last_twc is None
                else np.linalg.norm(twc[:3, 3] - last_twc[:3, 3])
            )
            rotation = (
                np.inf
                if last_twc is None
                else rotation_angle(last_twc[:3, :3], twc[:3, :3])
            )
            if translation < 0.04 and rotation < np.deg2rad(4.0):
                continue

            depth = np.asarray(Image.open(depth_path), dtype=np.uint16)
            color = np.asarray(Image.open(color_path).convert("RGB"), dtype=np.uint8)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(color)),
                o3d.geometry.Image(np.ascontiguousarray(depth)),
                depth_scale=1000.0,
                depth_trunc=args.depth_trunc,
                convert_rgb_to_intensity=False,
            )
            volume.integrate(rgbd, intrinsic, tcw)
            last_twc = twc
            last_image_mtime = image_mtime
            keyframes += 1

            if keyframes % args.extract_every == 0:
                cloud = volume.extract_point_cloud()
                cloud = cloud.voxel_down_sample(args.voxel_size)
                points = np.asarray(cloud.points, dtype=np.float32)
                colors = np.asarray(cloud.colors)
                colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
                rr.log(
                    "world/tsdf_map",
                    rr.Points3D(
                        points,
                        colors=colors,
                        radii=args.voxel_size * 0.45,
                    ),
                )
                o3d.io.write_point_cloud(str(output_path), cloud)
                print(
                    f"keyframes={keyframes} tsdf_points={len(points)} "
                    f"saved={output_path}",
                    flush=True,
                )
    except KeyboardInterrupt:
        cloud = volume.extract_point_cloud()
        o3d.io.write_point_cloud(str(output_path), cloud)
        print(f"Saved final TSDF map to {output_path}", flush=True)


if __name__ == "__main__":
    main()
