"""Strict, shared stereo-IR input preparation."""

from __future__ import annotations

import torch

Tensor = torch.Tensor

IMAGENET_MEAN_0_255 = (123.675, 116.28, 103.53)
IMAGENET_STD_0_255 = (58.395, 57.12, 57.375)


def prepare_ir_batch(
    value: object,
    *,
    name: str,
    height: int,
    width: int,
    device: torch.device,
) -> Tensor:
    """Validate one raw IR image and return ``[1,3,H,W]`` float32 0..255.

    A grayscale image is replicated to three channels.  No resize, padding,
    implicit normalization, CPU round-trip, or color conversion is performed.
    """

    image = torch.as_tensor(value, device=device)
    if image.ndim == 2:
        if tuple(image.shape) != (height, width):
            raise ValueError(
                f"{name} shape {tuple(image.shape)} does not match required "
                f"(height,width)=({height},{width})"
            )
        image = image.unsqueeze(-1)
    elif image.ndim == 3:
        if tuple(image.shape[:2]) != (height, width):
            raise ValueError(
                f"{name} shape {tuple(image.shape)} does not match required "
                f"(height,width)=({height},{width},C)"
            )
        if image.shape[-1] not in (1, 3):
            raise ValueError(f"{name} must have one or three channels, got {tuple(image.shape)}")
    else:
        raise ValueError(f"{name} must have shape HxW or HxWxC, got {tuple(image.shape)}")

    image = image.to(dtype=torch.float32)
    if not bool(torch.isfinite(image).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    if bool((image < 0).any()) or bool((image > 255).any()):
        raise ValueError(f"{name} must contain raw 0..255 pixels")
    if image.shape[-1] == 1:
        image = image.expand(-1, -1, 3)
    return image.permute(2, 0, 1).unsqueeze(0).contiguous()


def imagenet_normalize_0_255(image: Tensor) -> Tensor:
    """Apply the FFS ImageNet normalization to a raw model input."""

    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"Expected [B,3,H,W], got {tuple(image.shape)}")
    mean = torch.tensor(IMAGENET_MEAN_0_255, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD_0_255, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    return (image - mean) / std


def normalize_disparity_output(output: Tensor, *, height: int, width: int, device: torch.device) -> Tensor:
    """Normalize backend output to a full-resolution ``[H,W]`` float32 tensor."""

    disparity = output
    if disparity.ndim == 4 and tuple(disparity.shape[:2]) == (1, 1):
        disparity = disparity[0, 0]
    elif disparity.ndim == 3 and disparity.shape[0] == 1:
        disparity = disparity[0]
    if disparity.ndim != 2 or tuple(disparity.shape) != (height, width):
        raise ValueError(
            f"Backend disparity must be full-resolution {(height, width)}, got {tuple(output.shape)}"
        )
    return disparity.to(device=device, dtype=torch.float32)
