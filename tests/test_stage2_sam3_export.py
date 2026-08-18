"""Contract checks for the optional official-SAM3 sidecar adapter."""

from __future__ import annotations

import numpy as np

from tools.stage2.export_sam3_masks import (
    _install_init_state_compatibility,
    _records_from_outputs,
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
