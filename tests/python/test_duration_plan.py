import json
from pathlib import Path

import pytest

from minimax_music3_webgpu.duration_plan import plan_duration, plan_retained_frames


FIXTURE = Path(__file__).parents[1] / "fixtures" / "duration-plans.json"


def test_matches_hand_derived_official_plans():
    for entry in json.loads(FIXTURE.read_text())["cases"]:
        actual = plan_duration(
            duration_seconds=entry["durationSeconds"],
            prompt_tokens=entry["promptTokens"],
        )
        assert json.dumps(actual, separators=(",", ":")) == json.dumps(entry["plan"], separators=(",", ":"))


def test_floors_arbitrary_durations_to_frames_and_derives_exact_output_sizes():
    for duration_seconds, retained_frames, samples_per_channel, wav_bytes in (
        (5.04, 126, 222_208, 888_876),
        (6, 150, 264_192, 1_056_812),
        (10.5, 262, 462_336, 1_849_388),
    ):
        actual = plan_duration(duration_seconds=duration_seconds, prompt_tokens=0)
        assert actual["retainedFrames"] == retained_frames
        assert actual["samplesPerChannel"] == samples_per_channel
        assert actual["wavBytes"] == wav_bytes


def test_rejects_invalid_durations_and_durations_shorter_than_one_frame():
    for duration_seconds in (0, -1, float("nan"), float("inf"), 300.01):
        with pytest.raises(ValueError, match="greater than zero and at most 300"):
            plan_duration(duration_seconds=duration_seconds, prompt_tokens=0)
    with pytest.raises(ValueError, match="at least one frame"):
        plan_duration(duration_seconds=0.039, prompt_tokens=0)
    assert plan_duration(duration_seconds=0.04, prompt_tokens=0)["retainedFrames"] == 1


def test_rejects_the_removed_diagnostic_duration_keyword():
    with pytest.raises(TypeError, match="allow_diagnostic_duration"):
        plan_duration(duration_seconds=6, prompt_tokens=0, allow_diagnostic_duration=True)


def test_rejects_requests_that_exceed_the_language_model_context():
    with pytest.raises(ValueError, match="10240"):
        plan_duration(duration_seconds=300, prompt_tokens=2741)
    assert plan_retained_frames(retained_frames=7500, prompt_tokens=2740, termination="max-frames")["retainedFrames"] == 7500
    with pytest.raises(ValueError, match="10240"):
        plan_retained_frames(retained_frames=7500, prompt_tokens=2741, termination="max-frames")


def test_plans_actual_retained_frames_independently_of_the_requested_duration():
    for entry in json.loads(FIXTURE.read_text())["retainedFrameCases"]:
        actual = plan_retained_frames(retained_frames=entry["retainedFrames"], prompt_tokens=entry["promptTokens"], termination=entry["termination"])
        assert json.dumps(actual, separators=(",", ":")) == json.dumps(entry["plan"], separators=(",", ":"))


def test_exposes_max_frame_counts_separately_from_a_natural_end():
    actual = plan_retained_frames(retained_frames=201, prompt_tokens=40, termination="max-frames")
    assert actual["termination"] == "max-frames"
    assert actual["semanticDecisions"] == 202
    assert actual["rvqCalls"] == 1414
    assert actual["feedbackCalls"] == 201


def test_uses_explicit_flow_steps_while_preserving_the_30_step_default():
    assert plan_retained_frames(
        retained_frames=201,
        prompt_tokens=40,
        termination="max-frames",
        flow_steps=12,
    )["flowCalls"] == 24
    assert plan_duration(duration_seconds=10, prompt_tokens=40, flow_steps=12)["flowCalls"] == 24
    assert plan_duration(duration_seconds=10, prompt_tokens=40)["flowCalls"] == 60


def test_rejects_flow_steps_that_are_not_positive_safe_integers():
    for flow_steps in (0, -1, 1.5, 2**53):
        with pytest.raises(ValueError, match="positive safe integer"):
            plan_duration(duration_seconds=5, prompt_tokens=0, flow_steps=flow_steps)
        with pytest.raises(ValueError, match="positive safe integer"):
            plan_retained_frames(
                retained_frames=1,
                prompt_tokens=0,
                termination="max-frames",
                flow_steps=flow_steps,
            )


def test_accepts_every_product_duration_step():
    for duration_seconds in range(5, 301, 5):
        assert plan_duration(duration_seconds=duration_seconds, prompt_tokens=0)["retainedFrames"] == duration_seconds * 25


def test_rejects_non_integer_negative_and_non_boolean_runtime_inputs():
    with pytest.raises(ValueError, match="number"):
        plan_duration(duration_seconds=True, prompt_tokens=0)
    for kwargs in (
        {"duration_seconds": 5, "prompt_tokens": -1},
        {"duration_seconds": 5, "prompt_tokens": True},
    ):
        with pytest.raises(ValueError, match="integers"):
            plan_duration(**kwargs)
    for kwargs in (
        {"retained_frames": 1.5, "prompt_tokens": 0, "termination": "max-frames"},
        {"retained_frames": True, "prompt_tokens": 0, "termination": "max-frames"},
        {"retained_frames": -1, "prompt_tokens": 0, "termination": "max-frames"},
        {"retained_frames": 1, "prompt_tokens": 0.5, "termination": "max-frames"},
        {"retained_frames": 1, "prompt_tokens": True, "termination": "max-frames"},
    ):
        with pytest.raises(ValueError, match="integers"):
            plan_retained_frames(**kwargs)
    for retained_frames in (0, 7501):
        with pytest.raises(ValueError, match="between 1 and 7500"):
            plan_retained_frames(retained_frames=retained_frames, prompt_tokens=0, termination="max-frames")
    for termination in (1, "other"):
        with pytest.raises(ValueError, match="termination"):
            plan_retained_frames(retained_frames=1, prompt_tokens=0, termination=termination)
    with pytest.raises(TypeError, match="termination"):
        plan_retained_frames(retained_frames=1, prompt_tokens=0)
