"""Adapters from CameraRig single-camera detections to PCB rig observations."""

from __future__ import annotations

from typing import Any

from pointcloud_builder.rig_calibration.types import RigTargetObservation


def from_camera_rig_target_observation(
    observation: Any,
    *,
    observation_id: str,
    camera_id: str,
    pose_id: str,
    timestamp_ns: int | None,
    split: str = "solve",
) -> RigTargetObservation:
    """Copy one CameraRig detection without giving CameraRig multi-camera awareness."""

    quality = observation.quality.to_dict()
    quality["plugin_name"] = observation.plugin_name
    quality["target_frame"] = observation.target_frame
    quality["metadata"] = dict(observation.metadata)
    return RigTargetObservation(
        observation_id=observation_id,
        camera_id=camera_id,
        pose_id=pose_id,
        point_ids=observation.point_ids,
        object_points_m=observation.object_points_m,
        image_points_px=observation.image_points_px,
        timestamp_ns=timestamp_ns,
        quality=quality,
        split=split,
    )
