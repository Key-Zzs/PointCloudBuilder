"""Concurrent live acquisition and shared M6 rig processing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import queue
import threading
import time
from types import SimpleNamespace
from typing import Any

from camera_rig.api import (
    load_camera_config,
    load_provisioned_camera_bundle,
)

from pointcloud_builder.config import SamplingConfig, load_config
from pointcloud_builder.integrations.camera_rig import (
    create_ffs_builder,
    create_native_builder,
)
from pointcloud_builder.mapping.depth_packet import provision_identity_sha256
from pointcloud_builder.rig.config import RigConfig, RigLiveConfig, RigTimingConfig
from pointcloud_builder.rig.live_buffer import create_live_frame_buffers
from pointcloud_builder.rig.live_matcher import LiveRigFrameMatcher
from pointcloud_builder.rig.live_telemetry import LiveRigTelemetry
from pointcloud_builder.rig.live_worker import LiveCameraWorker, LiveWorkerError
from pointcloud_builder.rig.pipeline import RigCameraRuntime
from pointcloud_builder.rig.processor import RigFrameProcessor
from pointcloud_builder.rig.types import RigBuildResult, RigFrameSet
from pointcloud_builder.rig.validation import validate_rig_runtimes
from pointcloud_builder.workspace import SingleCameraWorkspacePipeline


class LiveRigWorkerFailure(RuntimeError):
    def __init__(self, errors: tuple[LiveWorkerError, ...]) -> None:
        self.errors = errors
        detail = "; ".join(
            f"{item.camera_name}:{item.phase}:{item.error_type}: {item.message}"
            for item in errors
        )
        super().__init__(f"one or more live camera workers failed: {detail}")


class LiveRigAcquisition:
    """Coordinate workers, bounded buffers, and host-clock matching."""

    def __init__(
        self,
        camera_configs: dict[str, Any],
        *,
        timing: RigTimingConfig,
        live_config: RigLiveConfig | None = None,
        session_factories: dict[str, Any] | None = None,
        required_streams_by_camera: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        if len(camera_configs) < 2:
            raise ValueError("live rig acquisition requires at least two cameras")
        if timing.mode != "nearest_host_timestamp":
            raise ValueError(
                "live rig acquisition requires nearest_host_timestamp timing"
            )
        self.camera_configs = dict(camera_configs)
        self.camera_names = tuple(sorted(camera_configs))
        self.timing = timing
        self.live_config = live_config or RigLiveConfig()
        self.stop_event = threading.Event()
        self.error_queue: queue.Queue[LiveWorkerError] = queue.Queue()
        self.telemetry = LiveRigTelemetry(self.camera_names)
        self.buffers = create_live_frame_buffers(
            self.camera_names, capacity=self.live_config.buffer_capacity
        )
        reference = timing.reference_camera or self.camera_names[0]
        self.matcher = LiveRigFrameMatcher(
            self.buffers,
            reference_camera=reference,
            maximum_skew_ms=timing.maximum_skew_ms,
            telemetry_history_capacity=self.live_config.telemetry_history_capacity,
        )
        factories = session_factories or {}
        required = required_streams_by_camera or {}
        self.workers = {
            name: LiveCameraWorker(
                name,
                self.camera_configs[name],
                buffer=self.buffers[name],
                stop_event=self.stop_event,
                error_queue=self.error_queue,
                telemetry=self.telemetry,
                required_streams=required.get(
                    name, ("color", "depth", "ir_left", "ir_right")
                ),
                session_factory=factories.get(name),
            )
            for name in self.camera_names
        }
        self._errors: list[LiveWorkerError] = []
        self._started = False
        self._stopped = False

    @property
    def started(self) -> bool:
        return self._started

    @property
    def stopped(self) -> bool:
        return self._stopped

    def start(self, *, readiness_timeout_s: float = 30.0) -> None:
        if self._started:
            raise RuntimeError("live rig acquisition has already started")
        if readiness_timeout_s <= 0:
            raise ValueError("readiness_timeout_s must be positive")
        self._started = True
        for worker in self.workers.values():
            worker.start()
        deadline = time.monotonic() + readiness_timeout_s
        for worker in self.workers.values():
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not worker.ready_event.wait(timeout=remaining):
                self.stop_event.set()
                for buffer in self.buffers.values():
                    buffer.notify_all()
                self._stop_workers()
                raise TimeoutError("live camera workers did not become ready in time")
        errors = self._drain_errors()
        if errors:
            self._stop_workers()
            raise LiveRigWorkerFailure(errors)

    def next_frame_set(self, timeout_s: float | None = None) -> RigFrameSet | None:
        if not self._started or self._stopped:
            raise RuntimeError("live rig acquisition is not running")
        errors = self._drain_errors()
        if errors:
            raise LiveRigWorkerFailure(errors)
        frame_set = self.matcher.match_next(
            self.live_config.matcher_wait_timeout_s if timeout_s is None else timeout_s
        )
        errors = self._drain_errors()
        if errors:
            raise LiveRigWorkerFailure(errors)
        return frame_set

    def stop(self) -> None:
        if not self._started or self._stopped:
            return
        self._stop_workers()

    def _stop_workers(self) -> None:
        self.stop_event.set()
        for buffer in self.buffers.values():
            buffer.notify_all()
        for worker in self.workers.values():
            worker.join(timeout=self.live_config.join_timeout_s)
        alive = [name for name, worker in self.workers.items() if worker.is_alive]
        if alive:
            for buffer in self.buffers.values():
                buffer.notify_all()
            for name in alive:
                self.workers[name].join(timeout=2.0)
            alive = [name for name, worker in self.workers.items() if worker.is_alive]
        if not alive:
            for buffer in self.buffers.values():
                buffer.close()
        self._drain_errors()
        self._stopped = True
        if alive:
            raise RuntimeError(f"live camera workers failed to join: {alive}")

    def _drain_errors(self) -> tuple[LiveWorkerError, ...]:
        new: list[LiveWorkerError] = []
        while True:
            try:
                error = self.error_queue.get_nowait()
            except queue.Empty:
                break
            self._errors.append(error)
            new.append(error)
        return tuple(new)

    def report(self) -> dict[str, Any]:
        self._drain_errors()
        matcher = asdict(self.matcher.stats())
        return {
            "schema_version": "pointcloud-builder.live-rig-acquisition.v1",
            "started": self._started,
            "stopped": self._stopped,
            "cameras": self.telemetry.report(),
            "buffers": {
                name: asdict(buffer.stats()) for name, buffer in self.buffers.items()
            },
            "matcher": matcher,
            "workers_alive": [
                name for name, worker in self.workers.items() if worker.is_alive
            ],
            "worker_errors": [error.summary() for error in self._errors],
        }

    def worker_errors(self) -> tuple[LiveWorkerError, ...]:
        self._drain_errors()
        return tuple(self._errors)

    def __enter__(self) -> "LiveRigAcquisition":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()


@dataclass(frozen=True)
class LiveRigBuild:
    result: RigBuildResult
    match_wait_ms: float
    processing_ms: float
    total_ms: float


class LiveRigPipeline:
    """Match live frames, then process them sequentially through M6."""

    def __init__(
        self,
        acquisition: LiveRigAcquisition,
        processor: RigFrameProcessor,
    ) -> None:
        self.acquisition = acquisition
        self.processor = processor

    def capture_next(self, timeout_s: float | None = None) -> LiveRigBuild:
        total_start = time.perf_counter()
        match_start = total_start
        frame_set = self.acquisition.next_frame_set(timeout_s)
        match_wait_ms = (time.perf_counter() - match_start) * 1000.0
        if frame_set is None:
            raise TimeoutError(
                "no complete live rig frame set was matched before timeout"
            )
        processing_start = time.perf_counter()
        result = self.processor.process_frame_set(
            frame_set,
            frame_match_ms=match_wait_ms,
            total_start_s=total_start,
        )
        processing_ms = (time.perf_counter() - processing_start) * 1000.0
        return LiveRigBuild(
            result=result,
            match_wait_ms=match_wait_ms,
            processing_ms=processing_ms,
            total_ms=(time.perf_counter() - total_start) * 1000.0,
        )

    def __enter__(self) -> "LiveRigPipeline":
        self.acquisition.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.acquisition.stop()


def build_live_rig(
    config: RigConfig,
    *,
    device: str = "auto",
    session_factories: dict[str, Any] | None = None,
) -> LiveRigPipeline:
    """Build validated CameraRig live workers and one shared frame processor."""

    if any(
        camera.source.type != "camera_rig_live" for camera in config.enabled_cameras
    ):
        raise ValueError(
            "build_live_rig requires every enabled source to be camera_rig_live"
        )
    no_sampling = SamplingConfig(mode="stride", num_points=1, enabled=False)
    camera_configs: dict[str, Any] = {}
    runtimes: dict[str, RigCameraRuntime] = {}
    serials: set[str] = set()
    for camera in config.enabled_cameras:
        source = camera.source
        camera_config = load_camera_config(source.camera_config)
        bundle = load_provisioned_camera_bundle(source.provision_artifact)
        if camera_config.camera.name != camera.name:
            raise ValueError(f"camera {camera.name!r} runtime config identity mismatch")
        if bundle.device.camera_name != camera.name:
            raise ValueError(
                f"camera {camera.name!r} provision bundle identity mismatch"
            )
        if camera_config.camera.serial != bundle.device.serial:
            raise ValueError(
                f"camera {camera.name!r} runtime/provision serial mismatch"
            )
        if camera_config.camera.serial in serials:
            raise ValueError("live rig cameras must have distinct serial identities")
        serials.add(camera_config.camera.serial)
        if not camera_config.capture.copy_frames:
            raise ValueError("live rig requires CameraRig copy_frames=true")
        fixed = bundle.fixed_mount_calibration
        if fixed is None or fixed.parent_frame != config.output_frame:
            raise ValueError(
                f"camera {camera.name!r} provision parent frame differs from rig output"
            )
        if camera.depth.mode == "native":
            context = create_native_builder(
                bundle,
                camera_name=camera.name,
                device=device,
                crop=camera.local_crop,
                sampling=no_sampling,
                use_rgb=camera.pointcloud.use_rgb,
            )
        else:
            if camera.pipeline_config is None:
                raise ValueError(
                    f"camera {camera.name!r} FFS mode requires pipeline_config"
                )
            pipeline_config = load_config(camera.pipeline_config)
            if pipeline_config.depth_source.ffs is None:
                raise ValueError(
                    f"camera {camera.name!r} pipeline_config has no FFS section"
                )
            context = create_ffs_builder(
                bundle,
                ffs_config=pipeline_config.depth_source.ffs,
                device=device,
                crop=camera.local_crop,
                sampling=no_sampling,
                use_rgb=camera.pointcloud.use_rgb,
            )
        camera_configs[camera.name] = camera_config
        runtimes[camera.name] = RigCameraRuntime(
            source=SimpleNamespace(camera_name=camera.name),
            pipeline=SingleCameraWorkspacePipeline(
                context,
                workspace_crop=config.workspace_crop,
                provision_sha256=provision_identity_sha256(source.provision_artifact),
            ),
            provenance={
                "source_type": "camera_rig_live",
                "depth_mode": camera.depth.mode,
                "pointcloud_format": ("xyzrgb" if camera.pointcloud.use_rgb else "xyz"),
            },
        )
    validate_rig_runtimes(config, runtimes)
    acquisition = LiveRigAcquisition(
        camera_configs,
        timing=config.timing,
        live_config=config.live,
        session_factories=session_factories,
    )
    return LiveRigPipeline(acquisition, RigFrameProcessor(config, runtimes))
