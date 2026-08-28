"""PCB-owned multi-camera, multi-pose calibration."""

from pointcloud_builder.rig_calibration.artifact import (
    load_observations,
    load_solution,
    write_observations,
    write_solution,
)
from pointcloud_builder.rig_calibration.config import RigCalibrationConfig
from pointcloud_builder.rig_calibration.solver import solve_rig_calibration
from pointcloud_builder.rig_calibration.types import (
    RigCalibrationObservations,
    RigCalibrationSolution,
    RigTargetObservation,
)

__all__ = [
    "RigCalibrationConfig",
    "RigCalibrationObservations",
    "RigCalibrationSolution",
    "RigTargetObservation",
    "load_observations",
    "load_solution",
    "solve_rig_calibration",
    "write_observations",
    "write_solution",
]
