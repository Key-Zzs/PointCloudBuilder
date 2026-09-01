"""Small deterministic frame graph using only CameraRig's public transform type."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import numpy as np

from pointcloud_builder.integrations.camera_rig.dependencies import RigidTransform


class TransformResolutionError(ValueError):
    """Raised for ambiguous, conflicting, or disconnected frame geometry."""


def resolve_transform(
    transforms: Sequence[RigidTransform],
    source_frame: str,
    target_frame: str,
) -> RigidTransform:
    """Resolve deterministic ``T_target_from_source`` with inverse and multi-hop edges."""

    if not source_frame.strip() or not target_frame.strip():
        raise TransformResolutionError("source_frame and target_frame must be non-empty")
    _validate_unique_edges(transforms)
    if source_frame == target_frame:
        return RigidTransform.identity(source_frame)

    adjacency: dict[str, list[RigidTransform]] = {}
    for transform in transforms:
        adjacency.setdefault(transform.source_frame, []).append(transform)
        adjacency.setdefault(transform.target_frame, []).append(transform.inverse())
    for edges in adjacency.values():
        edges.sort(key=lambda edge: (edge.target_frame, edge.source_frame))

    identity = RigidTransform.identity(source_frame)
    queue = deque([source_frame])
    resolved = {source_frame: identity}
    while queue:
        frame = queue.popleft()
        T_frame_from_source = resolved[frame]
        for edge in adjacency.get(frame, ()):
            neighbor = edge.target_frame
            if neighbor == source_frame:
                cycle_matrix = edge.matrix @ T_frame_from_source.matrix
                if not np.allclose(cycle_matrix, identity.matrix, atol=1e-7, rtol=1e-7):
                    raise TransformResolutionError(
                        f"conflicting transform cycle returning to source frame {source_frame!r}"
                    )
                continue
            T_neighbor_from_source = _compose_stable(edge, T_frame_from_source)
            previous = resolved.get(neighbor)
            if previous is not None:
                if not np.allclose(
                    previous.matrix,
                    T_neighbor_from_source.matrix,
                    atol=1e-7,
                    rtol=1e-7,
                ):
                    raise TransformResolutionError(
                        "conflicting transform paths for "
                        f"{neighbor!r} from source frame {source_frame!r}"
                    )
                continue
            resolved[neighbor] = T_neighbor_from_source
            queue.append(neighbor)
    if target_frame in resolved:
        return resolved[target_frame]
    frames = sorted(adjacency)
    raise TransformResolutionError(
        f"no transform path from {source_frame!r} to {target_frame!r}; available frames={frames}"
    )


def _validate_unique_edges(transforms: Sequence[RigidTransform]) -> None:
    accepted: list[RigidTransform] = []
    for candidate in transforms:
        for existing in accepted:
            if (
                candidate.source_frame == existing.source_frame
                and candidate.target_frame == existing.target_frame
            ):
                label = "duplicate" if np.allclose(candidate.matrix, existing.matrix) else "conflicting"
                raise TransformResolutionError(
                    f"{label} transform edge {candidate.target_frame}<-{candidate.source_frame}"
                )
            if (
                candidate.source_frame == existing.target_frame
                and candidate.target_frame == existing.source_frame
            ):
                matches = np.allclose(candidate.matrix, existing.inverse().matrix)
                label = "duplicate reverse" if matches else "conflicting reverse"
                raise TransformResolutionError(
                    f"{label} transform edge {candidate.target_frame}<-{candidate.source_frame}"
                )
        accepted.append(candidate)


def _compose_stable(left: RigidTransform, right: RigidTransform) -> RigidTransform:
    """Compose accepted SE(3) edges without amplifying float32 rotation drift.

    Camera SDK rotations commonly originate as float32 values.  Each edge can
    satisfy CameraRig's rigid-transform tolerance while a transpose/inverse and
    a subsequent matrix product exceeds it by a few ulps.  Projecting only the
    composed rotation to the closest proper rotation keeps the graph on SO(3)
    without modifying the authoritative input artifacts.
    """

    if left.source_frame != right.target_frame:
        raise TransformResolutionError(
            "cannot compose transforms: "
            f"{left.source_frame!r} != {right.target_frame!r} at the intermediate frame"
        )
    matrix = np.asarray(left.matrix @ right.matrix, dtype=np.float64)
    u, _singular_values, vt = np.linalg.svd(matrix[:3, :3])
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    matrix[:3, :3] = rotation
    return RigidTransform(
        source_frame=right.source_frame,
        target_frame=left.target_frame,
        matrix=matrix,
    )
