"""Contract checks for the optional official-SAM3 sidecar adapter."""

from __future__ import annotations

import numpy as np
import pytest

from tools.stage2.export_sam3_masks import (
    _configure_memory_profile,
    _install_init_state_compatibility,
    _records_from_outputs,
    _remove_stale_incomplete_episode_outputs,
    _resolve_attention_backend,
)


def test_sam3_singleton_mask_channel_is_normalized_without_flattening() -> None:
    outputs = {
        "out_obj_ids": np.asarray([7]),
        "out_probs": np.asarray([0.9], dtype=np.float32),
        # The official video API commonly emits [N_objects, 1, H, W].
        "out_binary_masks": np.asarray([[[[0.0, 1.0], [1.0, 0.0]]]], dtype=np.float32),
    }
    records = _records_from_outputs(
        outputs=outputs,
        frame_index=12,
        episode_index=3,
        concept_id="blue_ring",
        prompt_value="blue ring",
        shape=(2, 2),
    )
    assert len(records) == 1
    assert records[0].track_id == "blue_ring:sam3:7"
    assert records[0].valid is True
    assert records[0].binary_mask.shape == (2, 2)
    assert records[0].binary_mask.tolist() == [[False, True], [True, False]]


def test_sam3_optional_false_init_state_argument_is_only_filtered_when_unsupported() -> None:
    class LegacyMultiplexModel:
        def init_state(self, *, resource_path: str, offload_video_to_cpu: bool = False) -> dict[str, object]:
            return {"resource_path": resource_path, "offload_video_to_cpu": offload_video_to_cpu}

    class Predictor:
        model = LegacyMultiplexModel()

    predictor = Predictor()
    assert _install_init_state_compatibility(predictor) == "FILTER_OPTIONAL_OFFLOAD_STATE_TO_CPU_FALSE"
    assert predictor.model.init_state(
        resource_path="/tmp/frames",
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
    ) == {"resource_path": "/tmp/frames", "offload_video_to_cpu": False}
    try:
        predictor.model.init_state(
            resource_path="/tmp/frames",
            offload_video_to_cpu=False,
            offload_state_to_cpu=True,
        )
    except RuntimeError as exc:
        assert "cannot be adapted safely" in str(exc)
    else:
        raise AssertionError("CPU offload must never be silently dropped")


def test_sam3_fa3_acceleration_is_selected_only_when_its_runtime_module_exists() -> None:
    assert _resolve_attention_backend(module_available=lambda _: None) == (False, "PYTORCH_ATTENTION_NO_FA3")
    assert _resolve_attention_backend(module_available=lambda _: object()) == (True, "FLASH_ATTENTION_3")


def test_sam3_memory_profile_limits_batching_and_removes_only_stale_temp_outputs(tmp_path) -> None:
    class Model:
        batched_grounding_batch_size = 16
        postprocess_batch_size = 16

    class Predictor:
        model = Model()

    predictor = Predictor()
    assert _configure_memory_profile(
        predictor,
        grounding_batch_size=1,
        offload_video_to_cpu=True,
    ) == {
        "grounding_batch_size": 1,
        "postprocess_batch_size": 1,
        "offload_video_to_cpu": True,
    }
    assert predictor.model.batched_grounding_batch_size == 1
    assert predictor.model.postprocess_batch_size == 1
    with pytest.raises(ValueError, match="positive"):
        _configure_memory_profile(predictor, grounding_batch_size=0, offload_video_to_cpu=True)
    episode_root = tmp_path / "episodes"
    temporary = episode_root / ".episode_000.zarr.incomplete-123"
    temporary.mkdir(parents=True)
    (temporary / "partial").write_text("derived", encoding="utf-8")
    published = episode_root / "episode_000.zarr"
    published.mkdir()
    assert _remove_stale_incomplete_episode_outputs(episode_root) == [temporary.name]
    assert not temporary.exists()
    assert published.exists()
