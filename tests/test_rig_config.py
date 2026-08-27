from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from pointcloud_builder.rig import load_rig_config, parse_rig_config


def _raw(names: tuple[str, ...] = ("camera_a", "camera_b")) -> dict:
    return {
        "schema_version": "pointcloud-builder.rig.v1",
        "output_frame": "workspace",
        "cameras": [
            {
                "name": name,
                "enabled": True,
                "source": {
                    "type": "synthetic",
                    "capture_artifact": f"synthetic://{name}",
                    "provision_artifact": f"synthetic://{name}/bundle",
                },
                "depth": {"mode": "native"},
                "pipeline_config": None,
                "local_crop": {"enabled": False},
            }
            for name in names
        ],
        "timing": {"mode": "exact_index", "maximum_skew_ms": 33.4},
        "workspace_crop": {
            "enabled": True,
            "x": [-0.75, 0.75],
            "y": [-0.6, 0.6],
            "z": [-0.01, 0.6],
        },
        "fusion": {"enabled": False},
        "sampling": {
            "enabled": True,
            "mode": "voxel_fps",
            "num_points": 1024,
            "voxel_size": 0.01,
            "deterministic": True,
            "seed": 7,
        },
    }


def _live_raw() -> dict:
    raw = _raw()
    for camera in raw["cameras"]:
        name = camera["name"]
        camera["source"] = {
            "type": "camera_rig_live",
            "camera_config": f"private/{name}/runtime.yaml",
            "provision_artifact": f"private/{name}/provision",
        }
    raw["timing"] = {
        "mode": "nearest_host_timestamp",
        "maximum_skew_ms": 33.4,
        "reference_camera": "camera_a",
    }
    raw["live"] = {
        "buffer_capacity": 2,
        "matcher_wait_timeout_s": 0.1,
        "join_timeout_s": 5.0,
        "telemetry_history_capacity": 10_000,
    }
    return raw


def test_rig_config_is_strict_and_versioned() -> None:
    config = parse_rig_config(_raw())
    assert config.schema_version == "pointcloud-builder.rig.v1"
    assert [camera.name for camera in config.enabled_cameras] == [
        "camera_a",
        "camera_b",
    ]
    assert config.output_frame == config.workspace_crop.frame == "workspace"
    assert all(not camera.pointcloud.use_rgb for camera in config.cameras)
    assert config.live is None


def test_per_camera_rgb_pointcloud_is_explicit_and_strict() -> None:
    raw = _raw()
    raw["cameras"][0]["pointcloud"] = {"use_rgb": True}
    config = parse_rig_config(raw)
    assert config.cameras[0].pointcloud.use_rgb is True
    assert config.cameras[1].pointcloud.use_rgb is False

    raw["cameras"][0]["pointcloud"]["unknown"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        parse_rig_config(raw)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda raw: raw.update({"unknown": 1}), "unknown fields"),
        (lambda raw: raw["cameras"][0].update({"unknown": 1}), "unknown fields"),
        (lambda raw: raw.update({"cameras": []}), "non-empty"),
        (lambda raw: raw["cameras"].append(deepcopy(raw["cameras"][0])), "duplicate"),
        (lambda raw: raw["fusion"].update({"unknown": True}), "unknown fields"),
    ],
)
def test_rig_config_rejects_invalid_contracts(mutator, message: str) -> None:
    raw = _raw()
    mutator(raw)
    with pytest.raises(ValueError, match=message):
        parse_rig_config(raw)


def test_live_source_union_resolves_only_camera_config_and_provision(
    tmp_path: Path,
) -> None:
    raw = _live_raw()
    config = parse_rig_config(raw, base_dir=tmp_path)
    source = config.cameras[0].source
    assert source.type == "camera_rig_live"
    assert source.camera_config == str(
        (tmp_path / "private/camera_a/runtime.yaml").resolve()
    )
    assert source.provision_artifact == str(
        (tmp_path / "private/camera_a/provision").resolve()
    )
    assert config.live is not None
    assert config.live.buffer_capacity == 2
    assert config.live.matcher_wait_timeout_s == 0.1
    assert config.live.join_timeout_s == 5.0
    assert config.live.telemetry_history_capacity == 10_000


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            {
                "type": "camera_rig_replay",
                "capture_artifact": "capture",
                "provision_artifact": "provision",
                "camera_config": "runtime.yaml",
            },
            "unknown fields",
        ),
        (
            {"type": "camera_rig_replay", "provision_artifact": "provision"},
            "capture_artifact must be a non-empty string",
        ),
        (
            {
                "type": "camera_rig_live",
                "camera_config": "runtime.yaml",
                "provision_artifact": "provision",
                "capture_artifact": "capture",
            },
            "unknown fields",
        ),
        (
            {"type": "camera_rig_live", "provision_artifact": "provision"},
            "camera_config must be a non-empty string",
        ),
        (
            {
                "type": "synthetic",
                "capture_artifact": "synthetic://camera_a",
                "provision_artifact": "synthetic://camera_a/bundle",
                "camera_config": "runtime.yaml",
            },
            "unknown fields",
        ),
    ],
)
def test_source_is_a_strict_discriminated_union(source: dict, message: str) -> None:
    raw = _live_raw() if source["type"] == "camera_rig_live" else _raw()
    raw["cameras"][0]["source"] = source
    if source["type"] != "camera_rig_live":
        raw.pop("live", None)
        raw["timing"]["mode"] = "exact_index"
    with pytest.raises(ValueError, match=message):
        parse_rig_config(raw)


def test_replay_source_contract_remains_compatible() -> None:
    raw = _raw(("camera_a",))
    raw["cameras"][0]["source"] = {
        "type": "camera_rig_replay",
        "capture_artifact": "capture",
        "provision_artifact": "provision",
    }
    source = parse_rig_config(raw).cameras[0].source
    assert source.type == "camera_rig_replay"
    assert source.capture_artifact == "capture"
    assert source.provision_artifact == "provision"


def test_live_rig_requires_host_timestamp_matching() -> None:
    raw = _live_raw()
    raw["timing"]["mode"] = "exact_index"
    with pytest.raises(ValueError, match="live rigs require"):
        parse_rig_config(raw)


def test_timing_reference_must_be_enabled() -> None:
    raw = _raw()
    raw["cameras"][1]["enabled"] = False
    raw["timing"]["reference_camera"] = "camera_b"
    with pytest.raises(ValueError, match="enabled camera"):
        parse_rig_config(raw)


def test_live_section_is_rejected_without_an_enabled_live_source() -> None:
    raw = _raw()
    raw["live"] = {}
    with pytest.raises(ValueError, match="enabled camera_rig_live"):
        parse_rig_config(raw)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("buffer_capacity", 0, "between 1 and 4"),
        ("buffer_capacity", 5, "between 1 and 4"),
        ("buffer_capacity", 2.0, "must be an integer"),
        ("buffer_capacity", True, "must be an integer"),
        ("matcher_wait_timeout_s", 0.0, "must be positive"),
        ("matcher_wait_timeout_s", float("nan"), "finite number"),
        ("matcher_wait_timeout_s", float("inf"), "finite number"),
        ("matcher_wait_timeout_s", True, "finite number"),
        ("join_timeout_s", 0.0, "must be positive"),
        ("join_timeout_s", float("-inf"), "finite number"),
        ("telemetry_history_capacity", 0, "between 1 and 100000"),
        ("telemetry_history_capacity", 100_001, "between 1 and 100000"),
        ("telemetry_history_capacity", 10.0, "must be an integer"),
        ("telemetry_history_capacity", False, "must be an integer"),
    ],
)
def test_live_coordinator_limits_are_strict_and_bounded(
    field: str, value: object, message: str
) -> None:
    raw = _live_raw()
    raw["live"][field] = value
    with pytest.raises(ValueError, match=message):
        parse_rig_config(raw)


def test_live_section_rejects_unknown_fields_and_supplies_defaults() -> None:
    raw = _live_raw()
    raw["live"] = {}
    config = parse_rig_config(raw)
    assert config.live is not None
    assert config.live.buffer_capacity == 2
    assert config.live.matcher_wait_timeout_s == 0.1
    assert config.live.join_timeout_s == 5.0
    assert config.live.telemetry_history_capacity == 10_000

    raw["live"]["unknown"] = 1
    with pytest.raises(ValueError, match="unknown fields"):
        parse_rig_config(raw)


def test_live_section_is_optional_for_live_sources() -> None:
    raw = _live_raw()
    raw.pop("live")
    config = parse_rig_config(raw)
    assert config.live is None


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("cameras", 0, "enabled"), "true"),
        (("cameras", 0, "pointcloud", "use_rgb"), 1),
        (("workspace_crop", "enabled"), 1),
        (("fusion", "enabled"), "false"),
        (("fusion", "deterministic"), 1),
        (("sampling", "enabled"), 1),
        (("sampling", "deterministic"), "true"),
        (("sampling", "num_points"), 1024.0),
        (("sampling", "stride"), True),
    ],
)
def test_boolean_and_integer_fields_are_not_coerced(path: tuple, value: object) -> None:
    raw = _raw()
    raw["cameras"][0]["pointcloud"] = {"use_rgb": False}
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match="must be an? (boolean|integer)"):
        parse_rig_config(raw)


def test_public_live_rig_example_loads_with_placeholder_paths() -> None:
    root = Path(__file__).parents[1]
    config = load_rig_config(root / "configs/mapping/live_rig_example.yaml")
    assert [camera.source.type for camera in config.enabled_cameras] == [
        "camera_rig_live",
        "camera_rig_live",
    ]
    assert config.timing.mode == "nearest_host_timestamp"
    assert config.live is not None
