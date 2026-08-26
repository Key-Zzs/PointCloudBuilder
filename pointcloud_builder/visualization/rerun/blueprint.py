"""Default Rerun blueprint construction, isolated behind the optional SDK boundary."""

from __future__ import annotations

from typing import Any


def default_blueprint(rr: Any) -> Any | None:
    """Return a compact workspace/images/metrics layout when blueprint APIs are available."""

    blueprint = getattr(rr, "blueprint", None)
    if blueprint is None:
        return None
    return blueprint.Blueprint(
        blueprint.Horizontal(
            blueprint.Spatial3DView(name="Workspace", origin="/world"),
            blueprint.Vertical(
                blueprint.Spatial2DView(name="Camera A", origin="/rig/camera_a/rgb"),
                blueprint.Spatial2DView(name="Camera B", origin="/rig/camera_b/rgb"),
                blueprint.TimeSeriesView(name="Metrics", origin="/metrics"),
            ),
            column_shares=[2, 1],
        ),
        collapse_panels=True,
    )
