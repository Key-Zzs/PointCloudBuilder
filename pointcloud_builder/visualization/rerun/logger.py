"""Rerun SDK logger used only inside the dedicated child process."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pointcloud_builder.visualization.rerun.blueprint import default_blueprint
from pointcloud_builder.visualization.rerun.dependencies import require_rerun
from pointcloud_builder.visualization.rerun.packet import VisualizationPacket


class RerunPacketLogger:
    """Translate strict packets into stable Rerun entities and timelines."""

    def __init__(
        self,
        *,
        application_id: str,
        spawn: bool,
        connect_url: str | None,
        record_path: str | None,
    ) -> None:
        self.rr = require_rerun()
        self.recording = self.rr.RecordingStream(application_id)
        blueprint = default_blueprint(self.rr)
        record_output: Path | None = None
        if record_path is not None:
            record_output = Path(record_path)
            record_output.parent.mkdir(parents=True, exist_ok=True)
        if spawn:
            self.recording.spawn(default_blueprint=blueprint)
        sinks = []
        if spawn:
            sinks.append(self.rr.GrpcSink())
        elif connect_url is not None:
            sinks.append(self.rr.GrpcSink(connect_url))
        if record_output is not None:
            sinks.append(self.rr.FileSink(record_output))
        # spawn() is enough for viewer-only mode. set_sinks() is required for
        # an explicit connection, file-only output, or a viewer+file tee.
        if not spawn or record_output is not None:
            self.recording.set_sinks(*sinks, default_blueprint=blueprint)
        self._last_static_revision: int | None = None
        self._log_static()

    def _log_static(self) -> None:
        rr = self.rr
        self.recording.log(
            "/world/workspace_axes",
            rr.Arrows3D(
                origins=np.zeros((3, 3)),
                vectors=np.eye(3) * 0.1,
                colors=np.asarray(((255, 0, 0), (0, 255, 0), (0, 0, 255))),
            ),
            static=True,
        )
        self.recording.log(
            "/world/target",
            rr.Boxes3D(centers=[[0.0, 0.0, 0.0]], sizes=[[0.01, 0.01, 0.001]]),
            static=True,
        )

    def log(self, packet: VisualizationPacket) -> None:
        rr = self.rr
        self.recording.set_time("matched_set_index", sequence=packet.matched_set_index)
        self.recording.set_time("host_time_seconds", duration=packet.host_time_seconds)
        for camera in packet.cameras:
            base = f"/rig/{camera.camera_name}"
            matrix = camera.T_workspace_from_color
            self.recording.log(
                base,
                rr.Transform3D(
                    translation=matrix[:3, 3],
                    mat3x3=matrix[:3, :3],
                    relation=rr.TransformRelation.ParentFromChild,
                ),
            )
            intrinsics = camera.color_intrinsics
            self.recording.log(
                f"{base}/rgb",
                rr.Pinhole(
                    image_from_camera=np.asarray(
                        [
                            [intrinsics.fx, 0.0, intrinsics.cx],
                            [0.0, intrinsics.fy, intrinsics.cy],
                            [0.0, 0.0, 1.0],
                        ]
                    ),
                    resolution=[intrinsics.width, intrinsics.height],
                    camera_xyz=rr.ViewCoordinates.RDF,
                ),
                rr.Image(camera.rgb_preview),
            )
            # Per-camera clouds are already expressed in workspace coordinates.
            # Cancel the camera-parent pose so these points are not transformed twice.
            inverse = np.linalg.inv(matrix)
            self.recording.log(
                f"{base}/cloud",
                rr.Transform3D(
                    translation=inverse[:3, 3],
                    mat3x3=inverse[:3, :3],
                    relation=rr.TransformRelation.ParentFromChild,
                ),
            )
            self._log_cloud(f"{base}/cloud", camera.workspace_cloud)
        self._log_cloud("/clouds/concatenated", packet.concatenated_cloud)
        self._log_cloud("/clouds/fused", packet.fused_cloud)
        self._log_cloud("/clouds/sampled", packet.sampled_cloud)
        if packet.map is not None:
            self._log_map(packet)
        for name, value in sorted(packet.metrics.items()):
            self.recording.log(f"/metrics/{name}", rr.Scalars([value]))

    def _log_cloud(self, path: str, cloud: np.ndarray) -> None:
        kwargs: dict[str, Any] = {}
        if cloud.shape[1] == 6:
            colors = cloud[:, 3:6]
            kwargs["colors"] = (
                np.clip(colors, 0.0, 1.0) * 255.0
                if np.issubdtype(colors.dtype, np.floating)
                else colors
            ).astype(np.uint8)
        self.recording.log(path, self.rr.Points3D(cloud[:, :3], **kwargs))

    def _log_map(self, packet: VisualizationPacket) -> None:
        rr = self.rr
        map_packet = packet.map
        assert map_packet is not None
        if map_packet.reset:
            for path in (
                "/map/tsdf_mesh",
                "/map/tsdf_points",
                "/map/tsdf_points_raw",
                "/map/tsdf_points_cropped",
                "/map/tsdf_points_sampled",
                "/map/dynamic_overlay",
            ):
                self.recording.log(path, rr.Clear(recursive=True))
            self._last_static_revision = None
        log_static = (
            map_packet.static_revision is not None
            and map_packet.static_revision != self._last_static_revision
        )
        if log_static and map_packet.tsdf_points is not None:
            self._log_cloud("/map/tsdf_points", map_packet.tsdf_points)
        if log_static and map_packet.tsdf_points_raw is not None:
            self._log_cloud("/map/tsdf_points_raw", map_packet.tsdf_points_raw)
        if log_static and map_packet.tsdf_points_cropped is not None:
            self._log_cloud(
                "/map/tsdf_points_cropped", map_packet.tsdf_points_cropped
            )
        if log_static and map_packet.tsdf_points_sampled is not None:
            self._log_cloud(
                "/map/tsdf_points_sampled", map_packet.tsdf_points_sampled
            )
        if log_static and map_packet.tsdf_mesh is not None:
            mesh = map_packet.tsdf_mesh
            self.recording.log(
                "/map/tsdf_mesh",
                rr.Mesh3D(
                    vertex_positions=mesh.vertices,
                    triangle_indices=mesh.triangles,
                    vertex_colors=mesh.vertex_colors,
                ),
            )
        if log_static and (
            map_packet.tsdf_points is not None
            or map_packet.tsdf_points_raw is not None
            or map_packet.tsdf_points_cropped is not None
            or map_packet.tsdf_points_sampled is not None
            or map_packet.tsdf_mesh is not None
        ):
            self._last_static_revision = map_packet.static_revision
        if map_packet.dynamic_overlay is not None:
            self._log_cloud("/map/dynamic_overlay", map_packet.dynamic_overlay)
        if map_packet.raycast_depth is not None:
            self.recording.log(
                "/map/raycast_depth", rr.DepthImage(map_packet.raycast_depth)
            )
        if map_packet.dynamic_mask is not None:
            image = map_packet.dynamic_mask.astype(np.uint8) * 255
            self.recording.log("/map/dynamic_mask", rr.Image(image))

    def close(self) -> None:
        self.recording.flush(timeout_sec=30.0)
        self.recording.disconnect()
