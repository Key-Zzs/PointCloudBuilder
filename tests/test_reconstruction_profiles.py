from __future__ import annotations

from pathlib import Path

from pointcloud_builder.rig import load_rig_config


ROOT = Path(__file__).parents[1]
PROFILES = {
    "dense": ROOT / "configs/mapping/dense_rgb_reconstruction_example.yaml",
    "compact": ROOT / "configs/mapping/compact_rgb_reconstruction_example.yaml",
    "raw": ROOT / "configs/mapping/raw_rgb_concatenation_example.yaml",
}


def test_public_rgb_reconstruction_profiles_freeze_distinct_output_semantics() -> None:
    configs = {name: load_rig_config(path) for name, path in PROFILES.items()}
    assert all(
        camera.pointcloud.use_rgb
        for config in configs.values()
        for camera in config.enabled_cameras
    )
    assert configs["dense"].fusion.enabled
    assert configs["dense"].fusion.voxel_size_m == 0.0025
    assert not configs["dense"].sampling.enabled
    assert configs["compact"].fusion == configs["dense"].fusion
    assert configs["compact"].sampling.enabled
    assert configs["compact"].sampling.mode == "fps"
    assert configs["compact"].sampling.num_points == 30_000
    assert not configs["raw"].fusion.enabled
    assert not configs["raw"].sampling.enabled


def test_public_profiles_contain_only_portable_placeholder_paths() -> None:
    for path in PROFILES.values():
        text = path.read_text(encoding="utf-8")
        assert "/home/" not in text
        assert "serial:" not in text
        assert "REPLACE_WITH_" not in text or "serial" not in text.lower()
        assert "path/to/private/" in text


def test_no_public_fused_profile_defaults_to_redundant_voxel_sampling() -> None:
    for path in (ROOT / "configs/mapping").glob("*.yaml"):
        config = load_rig_config(path) if "rig" in path.name or "reconstruction" in path.name else None
        if config is None or not config.fusion.enabled or not config.sampling.enabled:
            continue
        assert config.sampling.mode not in {"voxel", "voxel_fps", "voxel_random"}, path
