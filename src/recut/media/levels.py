"""Short-time audio levels (10 ms RMS in dBFS) for silence-aware cut points and cut checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from recut.media.ffmpeg import SAMPLE_RATE

FRAME_SECONDS = 0.01
FLOOR_DB = -90.0


@dataclass
class Levels:
    db: npt.NDArray[np.float32]
    """One value per 10 ms frame."""
    frame: float = FRAME_SECONDS

    @classmethod
    def from_samples(cls, samples: npt.NDArray[np.float32], rate: int = SAMPLE_RATE) -> Levels:
        hop = int(rate * FRAME_SECONDS)
        n = len(samples) // hop
        if n == 0:
            return cls(np.zeros(0, dtype=np.float32))
        frames = samples[: n * hop].reshape(n, hop).astype(np.float64)
        rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
        db = np.maximum(20.0 * np.log10(rms), FLOOR_DB).astype(np.float32)
        return cls(db)

    @property
    def duration(self) -> float:
        return len(self.db) * self.frame

    def _index(self, t: float) -> int:
        return int(min(max(t / self.frame, 0), max(len(self.db) - 1, 0)))

    def at(self, t: float, window: float = 0.03) -> float:
        """Max level in a small window around t (a cut is only clean if the whole neighbourhood is quiet)."""
        if len(self.db) == 0:
            return FLOOR_DB
        lo, hi = self._index(t - window), self._index(t + window)
        return float(self.db[lo : hi + 1].max())

    def speech_level(self, start: float = 0.0, end: float | None = None) -> float:
        """The 90th percentile of the frame levels: a robust 'loud speech' reference."""
        if len(self.db) == 0:
            return FLOOR_DB
        lo = self._index(start)
        hi = self._index(end) if end is not None else len(self.db) - 1
        segment = self.db[lo : hi + 1]
        return float(np.percentile(segment, 90)) if len(segment) else FLOOR_DB

    def quietest(self, lo: float, hi: float, *, prefer: float | None = None) -> float:
        """The quietest 30 ms point in [lo, hi]; ties broken towards `prefer` (e.g. just after the last word)."""
        if hi <= lo or len(self.db) == 0:
            return lo
        a, b = self._index(lo), self._index(hi)
        if b <= a:
            return lo
        segment = self.db[a : b + 1].astype(np.float64)
        if len(segment) >= 3:
            smooth = np.convolve(segment, np.ones(3) / 3, mode="same")
        else:
            smooth = segment
        best = float(smooth.min())
        candidates = np.flatnonzero(smooth <= best + 1.5)
        target = self._index(prefer) - a if prefer is not None else (b - a) // 2
        choice = int(candidates[np.argmin(np.abs(candidates - target))])
        return round((a + choice) * self.frame, 3)
