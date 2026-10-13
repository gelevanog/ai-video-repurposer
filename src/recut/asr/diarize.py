"""Simple speaker diarization on CPU, with no model download: MFCC + pitch statistics per speech chunk,
agglomerative clustering, the number of speakers chosen by silhouette score.

It is deliberately light. It separates clearly different voices (a host and a guest, a man and a woman) well, and it
says "single speaker" when the voices are not separable instead of inventing speakers. It is not a substitute for a
neural diarizer (pyannote, NeMo) on overlapping speech or similar voices; the README reports where it fails."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.fft import dct
from scipy.spatial.distance import pdist, squareform

from recut.media.ffmpeg import SAMPLE_RATE
from recut.models import Transcript, Word

PAUSE_SPLIT = 0.35
MAX_CHUNK = 6.0
MIN_SILHOUETTE = 0.18
MIN_SHARE = 0.06


@dataclass
class Chunk:
    first: int
    last: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def chunk_words(words: list[Word], *, pause: float = PAUSE_SPLIT, max_len: float = MAX_CHUNK) -> list[Chunk]:
    chunks: list[Chunk] = []
    if not words:
        return chunks
    first = 0
    for i in range(1, len(words) + 1):
        end_here = i == len(words)
        if not end_here:
            gap = words[i].start - words[i - 1].end
            too_long = words[i].end - words[first].start > max_len
            end_here = gap >= pause or too_long
        if end_here:
            chunks.append(Chunk(first, i - 1, words[first].start, words[i - 1].end))
            first = i
    return chunks


def _mel_filterbank(n_mels: int, n_fft: int, rate: int) -> npt.NDArray[np.float64]:
    def hz_to_mel(f: npt.NDArray[np.float64] | float) -> npt.NDArray[np.float64]:
        return np.asarray(2595.0 * np.log10(1.0 + np.asarray(f) / 700.0), dtype=np.float64)

    def mel_to_hz(m: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.asarray(700.0 * (10.0 ** (m / 2595.0) - 1.0), dtype=np.float64)

    mels = np.linspace(hz_to_mel(60.0), hz_to_mel(rate / 2 * 0.95), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(mels) / rate).astype(int)
    bank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        lo, mid, hi = bins[m - 1], bins[m], bins[m + 1]
        for k in range(lo, mid):
            bank[m - 1, k] = (k - lo) / max(1, mid - lo)
        for k in range(mid, hi):
            bank[m - 1, k] = (hi - k) / max(1, hi - mid)
    return bank


_BANK = _mel_filterbank(40, 512, SAMPLE_RATE)


def chunk_features(samples: npt.NDArray[np.float32], rate: int = SAMPLE_RATE) -> npt.NDArray[np.float64] | None:
    """MFCC 1-19 mean and std over voiced frames, plus log pitch median and spread. None when too little speech."""
    win, hop, n_fft = int(0.025 * rate), int(0.010 * rate), 512
    if len(samples) < win * 4:
        return None
    n = 1 + (len(samples) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = samples[idx].astype(np.float64) * np.hamming(win)[None, :]
    energy = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    voiced = energy > max(np.percentile(energy, 35), 1e-4)
    if voiced.sum() < 8:
        return None
    frames = frames[voiced]
    spectrum = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2
    mel = np.log(spectrum @ _BANK.T + 1e-10)
    mfcc = dct(mel, type=2, axis=1, norm="ortho")[:, 1:20]
    pitch = _pitch(frames, rate)
    pitch = pitch[pitch > 0]
    log_pitch = np.log(pitch) if len(pitch) >= 4 else np.array([np.log(150.0)])
    return np.concatenate(
        [
            mfcc.mean(axis=0),
            mfcc.std(axis=0),
            [np.median(log_pitch) * 4.0, (np.percentile(log_pitch, 75) - np.percentile(log_pitch, 25)) * 2.0],
        ]
    )


def _pitch(frames: npt.NDArray[np.float64], rate: int) -> npt.NDArray[np.float64]:
    """Autocorrelation pitch per frame (60-400 Hz); 0 where the frame is not clearly periodic."""
    lo, hi = int(rate / 400), int(rate / 60)
    n_fft = 1024
    spec = np.fft.rfft(frames, n=n_fft)
    ac = np.fft.irfft(np.abs(spec) ** 2, n=n_fft)[:, : hi + 1]
    ac0 = ac[:, :1] + 1e-12
    norm = ac / ac0
    if norm.shape[1] <= lo:
        return np.zeros(len(frames))
    lags = np.argmax(norm[:, lo : hi + 1], axis=1) + lo
    peak = norm[np.arange(len(frames)), lags]
    out: npt.NDArray[np.float64] = np.where(peak > 0.45, rate / lags, 0.0)
    return out


def _silhouette(dist: npt.NDArray[np.float64], labels: npt.NDArray[np.int_]) -> float:
    unique = np.unique(labels)
    if len(unique) < 2:
        return -1.0
    scores = []
    for i in range(len(labels)):
        same = labels == labels[i]
        same[i] = False
        if not same.any():
            scores.append(0.0)
            continue
        a = dist[i, same].mean()
        b = min(dist[i, labels == other].mean() for other in unique if other != labels[i])
        scores.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(scores))


@dataclass
class DiarizationResult:
    labels: list[str]
    """One label per word."""
    speakers: list[str]
    method: str
    note: str
    silhouette: float


def diarize(
    transcript: Transcript, samples: npt.NDArray[np.float32], *, max_speakers: int = 4, rate: int = SAMPLE_RATE
) -> DiarizationResult:
    words = transcript.words
    single = DiarizationResult(["S1"] * len(words), ["S1"], "single", "", 0.0)
    chunks = chunk_words(words)
    feats: list[npt.NDArray[np.float64]] = []
    usable: list[int] = []
    for ci, chunk in enumerate(chunks):
        if chunk.duration < 0.6:
            continue
        a, b = int(chunk.start * rate), int(chunk.end * rate)
        f = chunk_features(samples[a:b], rate)
        if f is not None:
            feats.append(f)
            usable.append(ci)
    if len(feats) < 6:
        single.note = "too little speech to separate speakers; labelled as a single speaker"
        return single
    x = np.vstack(feats)
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-9)
    dist = squareform(pdist(x, metric="cosine"))
    tree = linkage(x, method="average", metric="cosine")
    best_k, best_s, best_labels = 1, -1.0, np.ones(len(x), dtype=int)
    durations = np.array([chunks[ci].duration for ci in usable])
    for k in range(2, max(2, max_speakers) + 1):
        labels = fcluster(tree, t=k, criterion="maxclust")
        shares = [durations[labels == c].sum() / durations.sum() for c in np.unique(labels)]
        if len(shares) < k or min(shares) < MIN_SHARE:
            continue
        s = _silhouette(dist, labels)
        if s > best_s + 0.02:
            best_k, best_s, best_labels = k, s, labels
    if best_k == 1 or best_s < MIN_SILHOUETTE:
        single.note = (
            "voices not separable with the light MFCC clustering "
            f"(best silhouette {max(best_s, 0):.2f}); labelled as a single speaker"
        )
        single.silhouette = max(best_s, 0.0)
        return single
    # Chunk labels, ordered by first appearance (S1 speaks first).
    chunk_label: dict[int, int] = dict(zip(usable, best_labels.tolist(), strict=True))
    for ci in range(len(chunks)):
        if ci not in chunk_label:
            chunk_label[ci] = _nearest_label(ci, chunk_label)
    _smooth(chunks, chunk_label)
    order: dict[int, str] = {}
    for ci in range(len(chunks)):
        raw = chunk_label[ci]
        if raw not in order:
            order[raw] = f"S{len(order) + 1}"
    labels_out = ["S1"] * len(words)
    for ci, chunk in enumerate(chunks):
        name = order[chunk_label[ci]]
        for wi in range(chunk.first, chunk.last + 1):
            labels_out[wi] = name
    return DiarizationResult(
        labels_out,
        list(order.values()),
        "mfcc-cluster",
        f"{len(order)} speakers separated by MFCC and pitch clustering (silhouette {best_s:.2f})",
        best_s,
    )


def _nearest_label(ci: int, known: dict[int, int]) -> int:
    for offset in range(1, 1000):
        for j in (ci - offset, ci + offset):
            if j in known:
                return known[j]
    return 1


def _smooth(chunks: list[Chunk], labels: dict[int, int]) -> None:
    """A short chunk between two chunks of the same speaker gets that speaker (MFCC stats are noisy below ~1 s)."""
    for ci in range(1, len(chunks) - 1):
        if chunks[ci].duration < 1.2 and labels[ci - 1] == labels[ci + 1] != labels[ci]:
            labels[ci] = labels[ci - 1]


def apply(transcript: Transcript, result: DiarizationResult) -> Transcript:
    words = [w.model_copy(update={"speaker": label}) for w, label in zip(transcript.words, result.labels, strict=True)]
    return transcript.model_copy(
        update={
            "words": words,
            "speakers": result.speakers,
            "diarization": result.method,
            "diarization_note": result.note,
        }
    )
