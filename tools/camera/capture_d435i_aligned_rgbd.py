import argparse
from pathlib import Path

import numpy as np
import pyrealsense2 as rs
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default=None)
    parser.add_argument("--width", type=int, default=424)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", default="captures/head_frame_000000.npz")
    parser.add_argument("--config-out", default="captures/head_aligned.yaml")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    cfg = rs.config()
    if args.serial:
        cfg.enable_device(args.serial)

    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
    cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    profile = pipeline.start(cfg)
    align = rs.align(rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())

    try:
        for _ in range(30):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to get aligned color/depth frame")

        rgb = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        color_profile = color_frame.profile.as_video_stream_profile()
        intr = color_profile.get_intrinsics()

        np.savez_compressed(
            out,
            rgb=rgb,
            depth=depth,
            rgb_timestamp=float(color_frame.get_timestamp()),
            depth_timestamp=float(depth_frame.get_timestamp()),
            depth_scale=depth_scale,
            width=intr.width,
            height=intr.height,
            fx=intr.fx,
            fy=intr.fy,
            cx=intr.ppx,
            cy=intr.ppy,
        )

        config = {
            "device": "cuda",
            "camera": {
                "name": "head",
                "aligned_depth_to_color": True,
                "depth_scale": depth_scale,
                "color_intrinsics": {
                    "width": intr.width,
                    "height": intr.height,
                    "fx": float(intr.fx),
                    "fy": float(intr.fy),
                    "cx": float(intr.ppx),
                    "cy": float(intr.ppy),
                },
                "depth_intrinsics": {
                    "width": intr.width,
                    "height": intr.height,
                    "fx": float(intr.fx),
                    "fy": float(intr.fy),
                    "cx": float(intr.ppx),
                    "cy": float(intr.ppy),
                },
            },
            "pointcloud": {
                "use_rgb": True,
                "output_format": "xyzrgb",
            },
        }

        config_path = Path(args.config_out)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        print(f"saved frame: {out}")
        print(f"saved config: {config_path}")
        print(f"rgb: {rgb.shape} {rgb.dtype}")
        print(f"depth: {depth.shape} {depth.dtype}")
        print(f"depth_scale: {depth_scale}")
        print(f"intrinsics: fx={intr.fx}, fy={intr.fy}, cx={intr.ppx}, cy={intr.ppy}")

    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()