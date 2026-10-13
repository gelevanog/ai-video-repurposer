"""Smart 9:16 framing on CPU: face detection on a few frames per second (OpenCV YuNet, a 230 KB ONNX model),
a smoothed "virtual camera" that follows the face, and a fallback when there is no face.

The camera moves only when the face leaves a dead zone, at a capped speed, and jumps (no pan) at shot changes, so the
result looks like an operator reframing, not a jittery tracker. Without enough faces (slides, screen recordings,
B-roll) the auto layout switches to "fit": the full 16:9 frame over a blurred fill."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from recut.logging_config import get_logger
from recut.models import CropKey, CropPlan

log = get_logger(__name__)

SAMPLE_FPS = 3.0
TRACK_MIN_FACE_RATIO = 0.35
DEAD_ZONE = 0.06
"""Fraction of the source width the face may drift before the camera follows."""
MAX_SPEED = 0.30
"""Camera speed cap, source widths per second."""
SCENE_CUT = 28.0
"""Mean absolute grey-level difference between samples that counts as a shot change."""


@dataclass
class Face:
    cx: float
    """0-1 of the frame width."""
    cy: float
    w: float
    """0-1 of the frame width."""
    score: float


class FaceDetector:
    def __init__(self, model: Path, *, threshold: float = 0.75) -> None:
        import cv2

        self._cv2 = cv2
        self.model = model
        self.threshold = threshold
        self._det = None
        self._size: tuple[int, int] | None = None

    def detect(self, frame: npt.NDArray[np.uint8]) -> list[Face]:
        h, w = frame.shape[:2]
        if self._det is None or self._size != (w, h):
            self._det = self._cv2.FaceDetectorYN.create(str(self.model), "", (w, h), self.threshold, 0.3, 20)
            self._size = (w, h)
        assert self._det is not None
        _, faces = self._det.detect(frame)
        out: list[Face] = []
        if faces is None:
            return out
        for row in faces:
            x, y, fw, fh = (float(v) for v in row[:4])
            score = float(row[-1])
            if fw < w * 0.025:  # tiny faces: audience, posters, logos
                continue
            out.append(Face(cx=(x + fw / 2) / w, cy=(y + fh / 2) / h, w=fw / w, score=score))
        return out


@dataclass
class Sample:
    t: float
    faces: list[Face]
    scene_cut: bool


def analyse(frames: Iterable[tuple[float, npt.NDArray[np.uint8]]], detector: FaceDetector) -> list[Sample]:
    samples: list[Sample] = []
    prev: npt.NDArray[np.float32] | None = None
    for t, frame in frames:
        grey = frame.mean(axis=2).astype(np.float32)[::4, ::4]
        cut = prev is not None and float(np.abs(grey - prev).mean()) > SCENE_CUT
        prev = grey
        samples.append(Sample(t=t, faces=detector.detect(frame), scene_cut=cut))
    return samples


def _target(faces: Sequence[Face], current: float | None) -> float | None:
    if not faces:
        return None
    largest = max(f.w for f in faces)
    big = [f for f in faces if f.w >= largest * 0.6]
    if current is None:
        return max(big, key=lambda f: f.w * f.score).cx
    # stay with the face nearest to where the camera is (no ping-pong between two speakers of similar size)
    return min(big, key=lambda f: abs(f.cx - current)).cx


def camera_path(samples: Sequence[Sample], *, window_frac: float) -> list[CropKey]:
    """Centre x (0-1) of the crop window per sample: dead zone, speed cap, jumps at shot changes, clamped so the
    window stays inside the frame."""
    half = window_frac / 2
    keys: list[CropKey] = []
    camera: float | None = None
    last_target: float | None = None
    targets: list[float | None] = []
    for s in samples:
        tgt = _target(s.faces, last_target if not s.scene_cut else None)
        if tgt is not None:
            last_target = tgt
        targets.append(tgt)
    # fill gaps: hold the last seen face; before the first face, use the first one
    first_known = next((t for t in targets if t is not None), 0.5)
    filled: list[float] = []
    hold = first_known
    for s, tgt in zip(samples, targets, strict=True):
        if s.scene_cut and tgt is None:
            hold = 0.5 if not any(t is not None for t in targets) else hold
        if tgt is not None:
            hold = tgt
        filled.append(hold)
    # median over 5 samples kills single-frame false detections
    med = [float(np.median(filled[max(0, i - 2) : i + 3])) for i in range(len(filled))]
    prev_t = samples[0].t if samples else 0.0
    for s, tgt in zip(samples, med, strict=True):
        dt = max(1e-3, s.t - prev_t)
        prev_t = s.t
        if camera is None or s.scene_cut:
            camera = tgt
        elif abs(tgt - camera) > DEAD_ZONE:
            step = min(abs(tgt - camera) - DEAD_ZONE / 2, MAX_SPEED * dt)
            camera += step if tgt > camera else -step
        camera = min(max(camera, half), 1 - half)
        keys.append(CropKey(t=round(s.t, 3), cx=round(camera, 4)))
    return simplify(keys)


def simplify(keys: Sequence[CropKey], tol: float = 0.004) -> list[CropKey]:
    """Drop keys that lie on the line between their neighbours (shorter ffmpeg expressions)."""
    if len(keys) <= 2:
        return list(keys)
    out = [keys[0]]
    for i in range(1, len(keys) - 1):
        a, b, c = out[-1], keys[i], keys[i + 1]
        if c.t - a.t <= 0:
            continue
        expected = a.cx + (c.cx - a.cx) * (b.t - a.t) / (c.t - a.t)
        jump = abs(c.cx - b.cx) > 0.05 and c.t - b.t < 0.5
        if abs(expected - b.cx) > tol or jump:
            out.append(b)
    out.append(keys[-1])
    return out


def plan(
    samples: Sequence[Sample], *, src_width: int, src_height: int, out_ratio: float = 9 / 16, layout: str = "auto"
) -> CropPlan:
    window_frac = min(1.0, src_height * out_ratio / max(1, src_width))
    with_face = sum(1 for s in samples if s.faces)
    ratio = with_face / len(samples) if samples else 0.0
    if layout == "center" or (layout == "auto" and window_frac >= 0.99):
        return CropPlan(layout="center", keys=[CropKey(t=0, cx=0.5)], face_ratio=ratio, samples=len(samples))
    if layout == "fit" or (layout == "auto" and ratio < TRACK_MIN_FACE_RATIO):
        note = "" if layout == "fit" else f"faces in {ratio:.0%} of sampled frames: full frame over a blurred fill"
        return CropPlan(layout="fit", keys=[], face_ratio=ratio, samples=len(samples), note=note)
    if with_face == 0:
        return CropPlan(
            layout="center", keys=[CropKey(t=0, cx=0.5)], face_ratio=0.0, samples=len(samples), note="no faces found"
        )
    keys = camera_path(samples, window_frac=window_frac)
    return CropPlan(
        layout="track",
        keys=keys,
        face_ratio=ratio,
        samples=len(samples),
        note=f"following the main face (found in {ratio:.0%} of sampled frames)",
    )


def x_expression(keys: Sequence[CropKey], *, src_width: int, crop_width: int) -> str:
    """ffmpeg crop `x` as a piecewise-linear function of t (evaluated per frame), in source pixels."""
    max_x = max(0, src_width - crop_width)

    def px(cx: float) -> float:
        return round(min(max(cx * src_width - crop_width / 2, 0), max_x), 1)

    if not keys:
        return str(px(0.5))
    if len(keys) == 1:
        return str(px(keys[0].cx))
    expr = str(px(keys[-1].cx))
    for a, b in reversed(list(zip(keys[:-1], keys[1:], strict=True))):
        xa, xb = px(a.cx), px(b.cx)
        span = max(1e-3, b.t - a.t)
        if abs(xa - xb) < 0.5 or (abs(xb - xa) > src_width * 0.05 and span < 0.5):
            seg = str(xa)  # hold, or a hard cut at the next key
        else:
            seg = f"{xa}+({xb - xa:.1f})*(t-{a.t:.3f})/{span:.3f}"
        expr = f"if(lt(t,{b.t:.3f}),{seg},{expr})"
    return f"if(lt(t,{keys[0].t:.3f}),{px(keys[0].cx)},{expr})"


def coverage(samples: Sequence[Sample], keys: Sequence[CropKey], *, window_frac: float) -> float:
    """Share of samples with a face whose main face centre lies inside the crop window (the framing metric)."""
    hits = total = 0
    for s in samples:
        if not s.faces:
            continue
        total += 1
        main = max(s.faces, key=lambda f: f.w)
        cx = interpolate(keys, s.t) if keys else 0.5
        if abs(main.cx - cx) <= window_frac / 2 - main.w / 4:
            hits += 1
    return hits / total if total else 1.0


def interpolate(keys: Sequence[CropKey], t: float) -> float:
    if not keys:
        return 0.5
    if t <= keys[0].t:
        return keys[0].cx
    for a, b in zip(keys[:-1], keys[1:], strict=True):
        if a.t <= t <= b.t:
            return a.cx + (b.cx - a.cx) * (t - a.t) / max(1e-6, b.t - a.t)
    return keys[-1].cx
