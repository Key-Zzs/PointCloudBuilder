from __future__ import annotations

import importlib.metadata
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

from pointcloud_builder.visualization.rerun.conversion import (
    _bounded_mesh,
    bounded_cloud,
    bounded_rgb_preview,
)
from pointcloud_builder.visualization.rerun.packet import (
    CameraVisualization,
    MapVisualization,
    PinholeVisualization,
    VisualizationPacket,
)
from pointcloud_builder.visualization.rerun.logger import RerunPacketLogger
from pointcloud_builder.visualization.rerun.process import (
    RerunOutputConfig,
    RerunViewerProcess,
    _carry_static_map,
)


def _packet(*, point_count: int = 8, index: int = 0) -> VisualizationPacket:
    cloud = np.column_stack(
        (
            np.linspace(0.0, 0.2, point_count, dtype=np.float32),
            np.zeros((point_count, 2), dtype=np.float32),
            np.ones((point_count, 3), dtype=np.float32),
        )
    )
    camera = CameraVisualization(
        camera_name="camera_a",
        rgb_preview=np.zeros((4, 6, 3), dtype=np.uint8),
        workspace_cloud=cloud,
        T_workspace_from_color=np.eye(4),
        color_intrinsics=PinholeVisualization(
            width=6,
            height=4,
            fx=4.0,
            fy=4.0,
            cx=3.0,
            cy=2.0,
        ),
    )
    return VisualizationPacket(
        matched_set_index=index,
        host_time_seconds=1.0 + index / 30.0,
        cameras=(camera,),
        concatenated_cloud=cloud,
        fused_cloud=cloud,
        sampled_cloud=cloud[:4],
        metrics={"processing_fps": 30.0, "fused_voxels": float(point_count)},
    )


def test_core_import_does_not_import_optional_rerun_sdk() -> None:
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, pointcloud_builder; assert 'rerun' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr


def test_packet_copies_cpu_arrays_and_rejects_non_rigid_or_private_payloads() -> None:
    packet = _packet()
    assert not packet.fused_cloud.flags.writeable
    source = np.eye(4)
    camera = packet.cameras[0]
    assert not camera.rgb_preview.flags.writeable
    source[0, 0] = 2.0
    assert camera.T_workspace_from_color[0, 0] == 1.0
    with pytest.raises(TypeError, match="NumPy"):
        CameraVisualization(
            camera_name="camera_a",
            rgb_preview=Path("private.png"),  # type: ignore[arg-type]
            workspace_cloud=camera.workspace_cloud,
            T_workspace_from_color=np.eye(4),
            color_intrinsics=camera.color_intrinsics,
        )
    reflected = np.eye(4)
    reflected[0, 0] = -1.0
    with pytest.raises(ValueError, match="determinant"):
        CameraVisualization(
            camera_name="camera_a",
            rgb_preview=camera.rgb_preview,
            workspace_cloud=camera.workspace_cloud,
            T_workspace_from_color=reflected,
            color_intrinsics=camera.color_intrinsics,
        )
    with pytest.raises(ValueError, match="bool"):
        MapVisualization(dynamic_mask=np.zeros((3, 3), dtype=np.uint8))


def test_visualization_downsampling_is_bounded_and_deterministic() -> None:
    points = torch.arange(600, dtype=torch.float32).reshape(100, 6)
    first = bounded_cloud(points, 17)
    second = bounded_cloud(points, 17)
    assert first.shape == (17, 6)
    assert np.array_equal(first, second)
    preview, stride = bounded_rgb_preview(np.zeros((20, 1001, 3), dtype=np.uint8))
    assert stride == 4
    assert preview.shape == (5, 251, 3)


def test_map_mesh_packet_is_bounded_and_indices_remain_valid() -> None:
    vertices = np.arange(1800, dtype=np.float32).reshape(600, 3)
    triangles = np.arange(1500, dtype=np.int64).reshape(500, 3) % len(vertices)
    mesh = _bounded_mesh(vertices, triangles, triangle_budget=37)
    assert mesh.triangles.shape == (37, 3)
    assert len(mesh.vertices) <= 111
    assert mesh.triangles.min() >= 0
    assert mesh.triangles.max() < len(mesh.vertices)


def test_latest_only_replacement_carries_unconsumed_static_map() -> None:
    static = MapVisualization(
        tsdf_points=np.zeros((3, 3), dtype=np.float32), static_revision=7
    )
    dynamic = MapVisualization(
        dynamic_overlay=np.ones((2, 3), dtype=np.float32), static_revision=7
    )
    dropped = VisualizationPacket(**{**_packet().__dict__, "map": static})
    replacement = VisualizationPacket(**{**_packet(index=1).__dict__, "map": dynamic})
    carried = _carry_static_map(dropped, replacement)
    assert carried.map is not None
    assert carried.map.tsdf_points is not None
    assert carried.map.dynamic_overlay is not None


def test_dynamic_only_packet_does_not_consume_static_revision() -> None:
    logger = object.__new__(RerunPacketLogger)
    logger.rr = object()
    logger.recording = object()
    logger._last_static_revision = None
    logged_paths: list[str] = []
    logger._log_cloud = lambda path, cloud: logged_paths.append(path)
    dynamic = VisualizationPacket(
        **{
            **_packet().__dict__,
            "map": MapVisualization(
                dynamic_overlay=np.ones((2, 3), dtype=np.float32),
                static_revision=8,
            ),
        }
    )
    logger._log_map(dynamic)
    assert logger._last_static_revision is None
    static = VisualizationPacket(
        **{
            **_packet(index=1).__dict__,
            "map": MapVisualization(
                tsdf_points=np.zeros((3, 3), dtype=np.float32),
                static_revision=8,
            ),
        }
    )
    logger._log_map(static)
    assert logger._last_static_revision == 8
    assert "/map/tsdf_points" in logged_paths


def test_output_modes_and_paths_fail_closed() -> None:
    with pytest.raises(ValueError, match="requires"):
        RerunOutputConfig()
    with pytest.raises(ValueError, match="mutually exclusive"):
        RerunOutputConfig(spawn=True, connect_url="rerun+http://127.0.0.1:9876/proxy")
    with pytest.raises(ValueError, match=".rrd"):
        RerunOutputConfig(record_path="recording.bin")


@pytest.mark.skipif(
    importlib.util.find_spec("rerun") is None,
    reason="rerun optional extra is not installed",
)
def test_headless_child_process_writes_rrd_and_closes(tmp_path: Path) -> None:
    assert importlib.metadata.version("rerun-sdk") == "0.36.3"
    output = tmp_path / "headless.rrd"
    viewer = RerunViewerProcess(
        RerunOutputConfig(record_path=str(output), queue_capacity=2)
    )
    viewer.start(timeout_s=30.0)
    assert viewer.submit(_packet())
    telemetry = viewer.close(timeout_s=30.0)
    assert telemetry.child_error is None
    assert telemetry.child_logged_packets == 1
    assert not telemetry.running
    assert output.stat().st_size > 0


@pytest.mark.skipif(
    importlib.util.find_spec("rerun") is None,
    reason="rerun optional extra is not installed",
)
def test_latest_only_queue_does_not_block_producer(tmp_path: Path) -> None:
    output = tmp_path / "latest-only.rrd"
    viewer = RerunViewerProcess(
        RerunOutputConfig(record_path=str(output), queue_capacity=1)
    )
    viewer.start(timeout_s=30.0)
    packet = _packet(point_count=20_000)
    durations_ms = []
    for index in range(40):
        start = time.perf_counter()
        viewer.submit(
            VisualizationPacket(
                matched_set_index=index,
                host_time_seconds=packet.host_time_seconds + index / 30.0,
                cameras=packet.cameras,
                concatenated_cloud=packet.concatenated_cloud,
                fused_cloud=packet.fused_cloud,
                sampled_cloud=packet.sampled_cloud,
                metrics=packet.metrics,
            )
        )
        durations_ms.append((time.perf_counter() - start) * 1000.0)
    telemetry = viewer.close(timeout_s=30.0)
    assert telemetry.produced_packets == 40
    assert telemetry.dropped_packets > 0
    assert telemetry.maximum_queue_depth <= 1
    # The first puts may start multiprocessing's local feeder thread. Evaluate
    # steady-state producer overhead after that bounded initialization warmup.
    assert np.quantile(durations_ms[5:], 0.95) <= 2.0
    assert telemetry.child_error is None
