from __future__ import annotations

import math

FRAMES_PER_SECOND = 25
MAX_PRODUCT_DURATION_SECONDS = 300
MAX_RETAINED_FRAMES = 7500
MAX_CONTEXT_TOKENS = 10_240
CHUNK_FRAMES = 200
CHUNK_HOP_FRAMES = 100
SAMPLES_PER_LATENT = 512
MAX_SAFE_INTEGER = 2**53 - 1


def _validate_non_negative_int(value: object, label: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be non-negative integers")


def _validate_flow_steps(value: object) -> None:
    if type(value) is not int or value < 1 or value > MAX_SAFE_INTEGER:
        raise ValueError("Flow steps must be a positive safe integer")


def _validate_duration(duration_seconds: float, prompt_tokens: int) -> None:
    _validate_non_negative_int(prompt_tokens, "Prompt tokens")
    if (
        type(duration_seconds) not in (int, float)
        or not math.isfinite(duration_seconds)
        or duration_seconds <= 0
        or duration_seconds > MAX_PRODUCT_DURATION_SECONDS
    ):
        raise ValueError("Duration seconds must be a finite number greater than zero and at most 300")
    if math.floor(duration_seconds * FRAMES_PER_SECOND) < 1:
        raise ValueError("Duration must produce at least one frame")


def plan_retained_frames(
    *,
    retained_frames: int,
    prompt_tokens: int,
    termination: str,
    flow_steps: int = 30,
) -> dict[str, object]:
    _validate_non_negative_int(retained_frames, "Retained frames")
    _validate_non_negative_int(prompt_tokens, "Prompt tokens")
    _validate_flow_steps(flow_steps)
    if retained_frames < 1 or retained_frames > MAX_RETAINED_FRAMES:
        raise ValueError("Retained frames must be between 1 and 7500")
    if termination not in ("max-frames", "natural-end"):
        raise ValueError("Invalid termination")
    if prompt_tokens + retained_frames > MAX_CONTEXT_TOKENS:
        raise ValueError(f"Prompt tokens plus frames must not exceed {MAX_CONTEXT_TOKENS}")
    chunk_count = 1 if retained_frames <= CHUNK_FRAMES else -(-retained_frames // CHUNK_HOP_FRAMES) - 1
    chunks: list[dict[str, int]] = []
    for index in range(chunk_count):
        start_frame = index * CHUNK_HOP_FRAMES
        frame_length = min(CHUNK_FRAMES, retained_frames - start_frame)
        latent_length = frame_length * 441 // 128
        crop_left_latents = 0 if index == 0 else 86
        crop_right_latents = 0 if index == chunk_count - 1 else 258
        chunks.append({
            "startFrame": start_frame,
            "frameLength": frame_length,
            "latentLength": latent_length,
            "cropLeftLatents": crop_left_latents,
            "cropRightLatents": crop_right_latents,
            "samplesPerChannel": (
                latent_length - crop_left_latents - crop_right_latents
            ) * SAMPLES_PER_LATENT,
        })
    samples_per_channel = sum(chunk["samplesPerChannel"] for chunk in chunks)
    semantic_decisions = retained_frames + (2 if termination == "natural-end" else 1)
    rvq_calls = (retained_frames + 1) * 7
    feedback_calls = retained_frames + (1 if termination == "natural-end" else 0)
    return {
        "retainedFrames": retained_frames,
        "termination": termination,
        "chunks": chunks,
        "samplesPerChannel": samples_per_channel,
        "wavBytes": 44 + samples_per_channel * 4,
        "flowCalls": chunk_count * flow_steps,
        "vocoderCalls": chunk_count * 2,
        "semanticDecisions": semantic_decisions,
        "rvqCalls": rvq_calls,
        "feedbackCalls": feedback_calls,
    }


def plan_duration(
    *,
    duration_seconds: float,
    prompt_tokens: int,
    flow_steps: int = 30,
) -> dict[str, object]:
    _validate_duration(duration_seconds, prompt_tokens)
    return plan_retained_frames(
        retained_frames=math.floor(duration_seconds * FRAMES_PER_SECOND),
        prompt_tokens=prompt_tokens,
        termination="max-frames",
        flow_steps=flow_steps,
    )
