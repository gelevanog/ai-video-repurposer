"""Rendering one vertical clip with a single ffmpeg pass: frame-accurate cut, 9:16 framing (tracked crop, centre
crop, full frame over a blurred fill, or an audiogram for audio-only sources), burned-in captions, fades and
loudness normalization to -14 LUFS (the level short-form platforms normalize to)."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from recut.config import Settings
from recut.logging_config import get_logger
from recut.media.ffmpeg import Probe, run, sample_frames
from recut.models import CaptionStyle, Clip, CropPlan, RenderInfo, Word
from recut.render import crop
from recut.render.captions import build_ass, clip_words

log = get_logger(__name__)


def _even(value: float) -> int:
    return max(2, int(round(value / 2.0)) * 2)


def escape_filter_path(path: Path) -> str:
    """A path inside a single-quoted ffmpeg filter argument."""
    text = str(path.resolve())
    return text.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")


@dataclass
class RenderJob:
    src: Path
    out: Path
    start: float
    end: float
    probe: Probe
    plan: CropPlan
    ass_path: Path
    width: int
    height: int


def video_filter(job: RenderJob, settings: Settings) -> tuple[list[str], str]:
    """Extra inputs and the filtergraph that produces [v]."""
    w, h = job.width, job.height
    subs = f"ass=filename='{escape_filter_path(job.ass_path)}':fontsdir='{escape_filter_path(settings.fonts_dir)}'"
    fps = "" if 0 < job.probe.fps <= 30.5 else "fps=30,"
    if not job.probe.has_video:
        duration = job.end - job.start
        inputs = ["-f", "lavfi", "-t", f"{duration:.3f}", "-i", f"color=c=0x15151f:s={w}x{h}:r=30"]
        graph = (
            f"[0:a]asplit=2[aw][ain];[aw]aformat=channel_layouts=mono,showwaves=s={w - 160}x{_even(h * 0.16)}:mode=cline:rate=30:"
            f"colors=0xA78BFA[wave];[1:v][wave]overlay=(W-w)/2:(H-h)/2-140,{subs},setsar=1[v]"
        )
        return inputs, graph
    sw, sh = job.probe.width, job.probe.height
    if job.plan.layout == "fit":
        fg_h = _even(w * sh / max(1, sw))
        offset = -120 if fg_h < h * 0.6 else 0
        graph = (
            f"[0:v]{fps}split=2[a][b];"
            f"[a]scale={_even(w / 4)}:{_even(h / 4)}:force_original_aspect_ratio=increase,crop={_even(w / 4)}:{_even(h / 4)},"
            f"boxblur=10:2,eq=brightness=-0.10:saturation=0.85,scale={w}:{h}[bg];"
            f"[b]scale={w}:{fg_h}:flags=lanczos[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2{offset:+d},{subs},setsar=1[v]"
        )
        return [], graph
    crop_w = min(sw, _even(sh * w / h))
    if job.plan.layout == "track" and job.plan.keys:
        x = crop.x_expression(job.plan.keys, src_width=sw, crop_width=crop_w)
    else:
        x = str(max(0, (sw - crop_w) // 2))
    graph = f"[0:v]{fps}crop=w={crop_w}:h={sh}:x='{x}':y=0,scale={w}:{h}:flags=lanczos,{subs},setsar=1[v]"
    return [], graph


def audio_filter(duration: float, settings: Settings, *, source: str = "[0:a]") -> str:
    fade_out = max(0.0, duration - 0.09)
    chain = f"{source}afade=t=in:st=0:d=0.04,afade=t=out:st={fade_out:.3f}:d=0.09"
    if settings.loudnorm:
        chain += ",loudnorm=I=-14:TP=-1.5:LRA=11"
    return chain + ",aresample=48000[a]"


def build_command(job: RenderJob, settings: Settings) -> list[str]:
    duration = job.end - job.start
    extra_inputs, vgraph = video_filter(job, settings)
    cmd = [
        settings.ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{job.start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(job.src),
        *extra_inputs,
        "-filter_complex",
        vgraph + ";" + audio_filter(duration, settings, source="[0:a]" if job.probe.has_video else "[ain]"),
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-c:v",
        "libx264",
        "-preset",
        settings.render_preset,
        "-crf",
        str(settings.render_crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
        "-shortest",
    ]
    if settings.render_threads:
        cmd += ["-threads", str(settings.render_threads)]
    cmd.append(str(job.out))
    return cmd


def plan_crop(src: Path, probe: Probe, start: float, end: float, settings: Settings, *, layout: str) -> CropPlan:
    if not probe.has_video:
        return CropPlan(layout="fit", note="audio-only source: audiogram")
    if layout in ("fit", "center"):
        return crop.plan([], src_width=probe.width, src_height=probe.height, layout=layout)
    detector = crop.FaceDetector(settings.face_model)
    frames = sample_frames(
        src,
        start=start,
        end=end,
        fps=crop.SAMPLE_FPS,
        width=min(640, probe.width),
        src_width=probe.width,
        src_height=probe.height,
        ffmpeg=settings.ffmpeg,
    )
    samples = crop.analyse(frames, detector)
    return crop.plan(samples, src_width=probe.width, src_height=probe.height, layout=layout)


def settings_hash(clip: Clip, words_digest: str, settings: Settings) -> str:
    material = {
        "start": clip.start,
        "end": clip.end,
        "style": clip.caption_style,
        "layout": clip.layout,
        "title": clip.title_card_text if clip.title_card else "",
        "overrides": clip.caption_overrides,
        "words": words_digest,
        "size": [settings.render_width, settings.render_height],
        "crf": settings.render_crf,
        "preset": settings.render_preset,
        "loudnorm": settings.loudnorm,
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True, default=str).encode()).hexdigest()[:16]


def render_clip(
    src: Path,
    probe: Probe,
    clip: Clip,
    words: list[Word],
    out: Path,
    settings: Settings,
    *,
    style: CaptionStyle | None = None,
    plan: CropPlan | None = None,
) -> RenderInfo:
    started = time.monotonic()
    style = style or clip.caption_style
    duration = clip.end - clip.start
    if plan is None:
        plan = plan_crop(src, probe, clip.start, clip.end, settings, layout=clip.layout)
    timed = clip_words(words, clip.start, clip.end, clip.caption_overrides)
    ass = build_ass(
        timed,
        style=style,
        duration=duration,
        layout=plan.layout,
        title=(clip.title_card_text or clip.title) if clip.title_card else "",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    ass_path = out.with_suffix(".ass")
    ass_path.write_text(ass, encoding="utf-8")
    tmp = out.with_name(out.stem + ".part.mp4")
    job = RenderJob(
        src, tmp, clip.start, clip.end, probe, plan, ass_path, settings.render_width, settings.render_height
    )
    run(build_command(job, settings), timeout=max(600.0, duration * 30))
    tmp.replace(out)
    elapsed = time.monotonic() - started
    digest = hashlib.sha256("|".join(f"{w.start}:{w.end}:{w.text}" for w in words[:5]).encode()).hexdigest()[:8]
    log.info("render.done", clip=clip.id, seconds=round(elapsed, 1), duration=round(duration, 1), layout=plan.layout)
    clip.crop = plan
    return RenderInfo(
        path=str(out),
        seconds=round(elapsed, 2),
        duration=round(duration, 3),
        width=settings.render_width,
        height=settings.render_height,
        style=style,
        layout=plan.layout,
        settings_hash=settings_hash(clip, digest, settings),
    )
