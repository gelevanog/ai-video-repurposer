from __future__ import annotations

import numpy as np

from recut.config import Settings
from recut.models import CropKey
from recut.render import crop
from recut.render.crop import Face, Sample, camera_path, coverage, interpolate, plan, simplify, x_expression


def _samples(xs: list[float | None], *, cut_at: int | None = None, fps: float = 3.0) -> list[Sample]:
    return [
        Sample(t=i / fps, faces=[Face(cx=x, cy=0.4, w=0.12, score=0.9)] if x is not None else [], scene_cut=i == cut_at)
        for i, x in enumerate(xs)
    ]


def _split_top(text: str) -> list[str]:
    parts, depth, cur = [], 0, ""
    for ch in text:
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
            continue
        depth += ch == "("
        depth -= ch == ")"
        cur += ch
    parts.append(cur)
    return parts


def _eval_expression(expr: str, t: float) -> float:
    """Evaluate the if(lt(t,T),A,B) subset of ffmpeg expressions that x_expression emits."""
    if expr.startswith("if(lt(t,"):
        inner = expr[len("if(") : -1]
        cond, a, b = _split_top(inner)
        threshold = float(cond[len("lt(t,") : -1])
        return _eval_expression(a, t) if t < threshold else _eval_expression(b, t)
    return float(eval(expr, {"t": t}))  # noqa: S307 - test-only, arithmetic built by our own code


def test_camera_stays_put_inside_the_dead_zone() -> None:
    keys = camera_path(_samples([0.50, 0.52, 0.49, 0.53, 0.51] * 6), window_frac=0.32)
    assert all(abs(k.cx - keys[0].cx) < 0.03 for k in keys)


def test_camera_follows_a_moving_face_at_a_capped_speed() -> None:
    xs = [0.3] * 6 + [0.7] * 30
    keys = camera_path(_samples(xs), window_frac=0.32)
    assert keys[0].cx < 0.35 and keys[-1].cx > 0.6
    path = [interpolate(keys, i / 3) for i in range(len(xs))]
    speeds = [abs(b - a) * 3 for a, b in zip(path, path[1:], strict=False)]
    assert max(speeds) <= crop.MAX_SPEED + 0.02


def test_camera_jumps_at_a_scene_cut() -> None:
    xs = [0.3] * 6 + [0.75] * 6
    keys = camera_path(_samples(xs, cut_at=6), window_frac=0.32)
    assert interpolate(keys, 6 / 3 + 0.01) > 0.6


def test_window_is_clamped_inside_the_frame() -> None:
    keys = camera_path(_samples([0.02] * 10 + [0.99] * 10, cut_at=10), window_frac=0.32)
    assert all(0.16 - 1e-6 <= k.cx <= 0.84 + 1e-6 for k in keys)


def test_a_single_false_detection_is_ignored() -> None:
    xs = [0.5] * 5 + [0.95] + [0.5] * 5
    keys = camera_path(_samples(xs), window_frac=0.32)
    assert all(abs(k.cx - 0.5) < 0.05 for k in keys)


def test_layout_decisions() -> None:
    faces = _samples([0.5] * 10)
    none = _samples([None] * 10)
    assert plan(faces, src_width=1280, src_height=720).layout == "track"
    assert plan(none, src_width=1280, src_height=720).layout == "fit"
    assert plan(faces, src_width=1280, src_height=720, layout="center").layout == "center"
    assert plan(faces, src_width=720, src_height=1280).layout == "center"  # already vertical
    assert plan(none, src_width=1280, src_height=720, layout="track").layout == "center"


def test_simplify_drops_collinear_keys() -> None:
    keys = [CropKey(t=i, cx=0.3 + 0.01 * i) for i in range(20)]
    assert len(simplify(keys)) == 2


def test_x_expression_matches_the_planned_path() -> None:
    keys = [CropKey(t=0, cx=0.3), CropKey(t=2, cx=0.3), CropKey(t=4, cx=0.6), CropKey(t=6, cx=0.6)]
    expr = x_expression(keys, src_width=1280, crop_width=404)
    for t in (0.0, 1.0, 3.0, 5.0, 7.0):
        expected = min(max(interpolate(keys, t) * 1280 - 202, 0), 1280 - 404)
        assert abs(_eval_expression(expr, t) - expected) < 1.0


def test_x_expression_for_a_fixed_window() -> None:
    assert x_expression([CropKey(t=0, cx=0.5)], src_width=1280, crop_width=404) == "438.0"
    assert x_expression([], src_width=1280, crop_width=404) == "438.0"


def test_coverage_compares_tracked_and_center() -> None:
    samples = _samples([0.2] * 10)
    tracked = [CropKey(t=0, cx=0.2)]
    assert coverage(samples, tracked, window_frac=0.32) == 1.0
    assert coverage(samples, [CropKey(t=0, cx=0.5)], window_frac=0.32) == 0.0


def test_face_detector_runs_on_a_blank_frame(settings: Settings) -> None:
    detector = crop.FaceDetector(settings.face_model)
    assert detector.detect(np.zeros((180, 320, 3), dtype=np.uint8)) == []
