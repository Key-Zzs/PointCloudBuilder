"""Contract checks for the optional official-SAM3 sidecar adapter."""

from __future__ import annotations

import numpy as np

from tools.stage2.export_sam3_masks import _records_from_outputs


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
