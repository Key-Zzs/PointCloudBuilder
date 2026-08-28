"""Connected bipartite camera/target-pose graph preflight."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass

from pointcloud_builder.rig_calibration.types import RigTargetObservation


class CalibrationPreflightError(ValueError):
    """A stable fail-closed preflight reason."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class ObservationGraphReport:
    camera_count: int
    pose_count: int
    edge_count: int
    connected: bool
    components: tuple[tuple[str, ...], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "camera_count": self.camera_count,
            "pose_count": self.pose_count,
            "edge_count": self.edge_count,
            "connected": self.connected,
            "components": [list(component) for component in self.components],
        }


def analyze_observation_graph(
    observations: Iterable[RigTargetObservation],
    *,
    camera_ids: Iterable[str],
    split: str = "solve",
) -> ObservationGraphReport:
    selected = tuple(item for item in observations if item.split == split)
    cameras = tuple(sorted(set(camera_ids)))
    poses = tuple(sorted({item.pose_id for item in selected}))
    adjacency: dict[str, set[str]] = defaultdict(set)
    edges: set[tuple[str, str]] = set()
    for item in selected:
        camera_node = f"camera:{item.camera_id}"
        pose_node = f"pose:{item.pose_id}"
        adjacency[camera_node].add(pose_node)
        adjacency[pose_node].add(camera_node)
        edges.add((item.camera_id, item.pose_id))
    nodes = {f"camera:{camera_id}" for camera_id in cameras} | {
        f"pose:{pose_id}" for pose_id in poses
    }
    components: list[tuple[str, ...]] = []
    remaining = set(nodes)
    while remaining:
        start = min(remaining)
        queue = deque((start,))
        visited: set[str] = set()
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            queue.extend(sorted(adjacency[node] - visited))
        remaining -= visited
        components.append(tuple(sorted(visited)))
    return ObservationGraphReport(
        camera_count=len(cameras),
        pose_count=len(poses),
        edge_count=len(edges),
        connected=len(components) == 1 and bool(nodes),
        components=tuple(sorted(components)),
    )


def require_connected_graph(report: ObservationGraphReport) -> None:
    if not report.connected:
        raise CalibrationPreflightError(
            "DISCONNECTED_CALIBRATION_GRAPH",
            f"observation graph has {len(report.components)} connected components",
        )
