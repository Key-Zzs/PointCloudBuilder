#!/usr/bin/env python3
"""Export SAM 3.1 Object-Multiplex video masks into compact atomic sidecars.

This runs only in the isolated ``paper-a-sam3`` environment.  It keeps every
LeRobot source input read-only, splits a source video into temporary complete
episode frame directories, and processes each episode as one SAM video session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np


PCB_ROOT = Path(__file__).resolve().parents[2]
if str(PCB_ROOT) not in sys.path:
    sys.path.insert(0, str(PCB_ROOT))

from pointcloud_builder.segmentation import ManifestPromptProvider, SegmentationSidecar  # noqa: E402
from pointcloud_builder.segmentation.types import InstanceMask, SegmentationProvenance  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path for path in root.rglob("*") if path.is_file()):
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(str(item.stat().st_size).encode("ascii"))
        digest.update(_sha256_file(item).encode("ascii"))
    return digest.hexdigest()


def _episode_ends(dataset_root: Path) -> np.ndarray:
    import zarr

    source = zarr.open_group(str(dataset_root / "sidecars" / "realsense.zarr"), mode="r")
    return np.asarray(source["meta/episode_ends"][:], dtype=np.int64)


def _video_paths(dataset_root: Path) -> list[Path]:
    paths = sorted((dataset_root / "videos" / "observation.images.head_rgb").glob("chunk-*/file-*.mp4"))
    if not paths:
        raise FileNotFoundError("head RGB videos are required for SAM3 sidecar export")
    return paths


def _iter_rgb_frames(paths: list[Path]) -> Iterator[np.ndarray]:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required in paper-a-sam3 to decode LeRobot RGB video") from exc
    for path in paths:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                yield frame.to_ndarray(format="rgb24")


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _records_from_outputs(
    *,
    outputs: dict[str, Any],
    frame_index: int,
    episode_index: int,
    concept_id: str,
    prompt_value: str,
    shape: tuple[int, int],
) -> list[InstanceMask]:
    ids = _as_numpy(outputs.get("out_obj_ids", []))
    scores = _as_numpy(outputs.get("out_probs", []))
    masks = _as_numpy(outputs.get("out_binary_masks", []))
    records: list[InstanceMask] = []
    for index, object_id in enumerate(ids.tolist()):
        mask = np.asarray(masks[index] > 0.5, dtype=bool)
        if mask.shape != shape:
            raise ValueError(f"SAM3 mask shape {mask.shape} does not match source RGB shape {shape}")
        records.append(InstanceMask(
            frame_index=frame_index,
            episode_index=episode_index,
            track_id=f"{concept_id}:sam3:{object_id}",
            concept_id=concept_id,
            binary_mask=mask,
            bbox=_bbox(mask),
            score=float(scores[index]) if index < len(scores) else float("nan"),
            prompt_type="text",
            prompt_value=prompt_value,
            valid=bool(mask.any()),
        ))
    if not records:
        records.append(InstanceMask(
            frame_index=frame_index,
            episode_index=episode_index,
            track_id=f"{concept_id}:sam3:none",
            concept_id=concept_id,
            binary_mask=np.zeros(shape, dtype=bool),
            bbox=(0, 0, 0, 0),
            score=float("nan"),
            prompt_type="text",
            prompt_value=prompt_value,
            valid=False,
        ))
    return records


def _run_concept_video(
    *,
    predictor: Any,
    frame_dir: Path,
    global_start: int,
    episode_index: int,
    concept_id: str,
    text: str,
    shape: tuple[int, int],
) -> Iterator[InstanceMask]:
    response = predictor.handle_request({"type": "start_session", "resource_path": str(frame_dir)})
    session_id = response["session_id"]
    try:
        initial = predictor.handle_request({"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": text})
        per_frame: dict[int, dict[str, Any]] = {0: initial["outputs"]}
        for streamed in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": session_id}):
            per_frame[int(streamed["frame_index"])] = streamed["outputs"]
        for local_index in range(len(list(frame_dir.glob("*.jpg")))):
            output = per_frame.get(local_index)
            if output is None:
                yield InstanceMask(
                    global_start + local_index, episode_index, f"{concept_id}:sam3:none", concept_id,
                    np.zeros(shape, dtype=bool), (0, 0, 0, 0), float("nan"), "text", text, False,
                )
                continue
            yield from _records_from_outputs(
                outputs=output, frame_index=global_start + local_index, episode_index=episode_index,
                concept_id=concept_id, prompt_value=text, shape=shape,
            )
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})


def _code_commit() -> str:
    import sam3

    source = Path(sam3.__file__).resolve().parent.parent
    try:
        return subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("SAM3 must be installed from an official Git checkout so its code commit is recorded") from exc


def run(args: argparse.Namespace) -> dict[str, Any]:
    from PIL import Image
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    dataset_root = Path(args.lerobot_path).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    prompts = ManifestPromptProvider.from_yaml(args.prompt_config)
    checkpoint = Path(args.checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM3.1 checkpoint unavailable: {checkpoint}")
    ends = _episode_ends(dataset_root)
    requested = set(range(len(ends))) if args.episodes is None else set(args.episodes)
    if not requested:
        raise ValueError("--episodes must name at least one episode when provided")
    if any(episode < 0 or episode >= len(ends) for episode in requested):
        raise ValueError(f"requested SAM3 episode is outside source range 0..{len(ends) - 1}")
    provenance = SegmentationProvenance(
        execution="sidecar",
        code_commit=_code_commit(),
        checkpoint_id=str(args.checkpoint_id),
        checkpoint_sha256=_sha256_file(checkpoint),
        prompt_config_sha256=prompts.config_hash,
        source_dataset_sha256=_tree_digest(dataset_root),
    )
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"output already exists; use --resume only for matching provenance: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest = {"schema_version": "paper_a_sam3_episode_sidecars_v1", "provenance": provenance.__dict__, "episodes": {}}
    if args.resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("provenance") != provenance.__dict__:
            raise ValueError("cannot reuse SAM sidecars: source/prompt/checkpoint/code provenance differs")
    predictor = build_sam3_multiplex_video_predictor(checkpoint_path=str(checkpoint))
    frame_iterator = _iter_rgb_frames(_video_paths(dataset_root))
    start = 0
    for episode, end_value in enumerate(ends.tolist()):
        end = int(end_value)
        if episode not in requested:
            # Keep the one-pass video decoder aligned with global source frame
            # indices while deliberately avoiding inference for non-pilot data.
            for _ in range(end - start):
                next(frame_iterator)
            start = end
            continue
        episode_name = f"episode_{episode:03d}.zarr"
        target = output_root / "episodes" / episode_name
        if target.exists() and str(episode) in manifest.get("episodes", {}):
            for _ in range(end - start):
                next(frame_iterator)
            start = end
            continue
        with tempfile.TemporaryDirectory(prefix=f"sam3_ep{episode:03d}_", dir=output_root) as temporary:
            frame_dir = Path(temporary)
            image_shape: tuple[int, int] | None = None
            for local_index in range(end - start):
                rgb = next(frame_iterator)
                image_shape = tuple(int(value) for value in rgb.shape[:2])
                Image.fromarray(rgb).save(frame_dir / f"{local_index:05d}.jpg", quality=95)
            if image_shape is None:
                raise RuntimeError(f"empty episode {episode} is not a valid SAM3 video")
            temporary_target = target.with_name(f".{target.name}.incomplete-{os.getpid()}")
            if temporary_target.exists():
                shutil.rmtree(temporary_target)
            with SegmentationSidecar.open_stream_writer(
                temporary_target,
                provenance=provenance,
                expected_frame_indices=range(start, end),
            ) as writer:
                for prompt in prompts.prompts:
                    for record in _run_concept_video(
                        predictor=predictor, frame_dir=frame_dir, global_start=start, episode_index=episode,
                        concept_id=prompt.concept_id, text=prompt.value, shape=image_shape,
                    ):
                        writer.add(record)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary_target.replace(target)
        manifest["episodes"][str(episode)] = {"path": str(target), "start": start, "end": end}
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        start = end
    if start != int(ends[-1]):
        raise RuntimeError("source video did not supply every indexed frame")
    return {"output": str(output_root), "episodes": sorted(requested), "provenance": provenance.__dict__}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot-path", required=True)
    parser.add_argument("--prompt-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--checkpoint-id", default="facebook/sam3.1:sam3.1_multiplex.pt")
    parser.add_argument("--episodes", nargs="+", type=int, help="Only infer these whole episodes (pilot gate)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
