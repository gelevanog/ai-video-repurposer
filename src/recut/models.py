"""The data model of a project: source media, transcript, sentences, clips, chapters and written outputs.

Everything is a pydantic model so the API, the MCP server, the CLI and project.json share one schema."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

ClipStatus = Literal["suggested", "approved", "rejected"]
CaptionStyle = Literal["bold", "boxed", "karaoke", "pop", "minimal"]
Layout = Literal["auto", "track", "fit", "center"]
StageName = Literal["probe", "audio", "transcribe", "diarize", "clips", "chapters", "posts"]
StageState = Literal["pending", "running", "done", "failed", "skipped"]


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Word(BaseModel):
    start: float
    end: float
    text: str
    """As transcribed, with punctuation attached ("world," / "Really?")."""
    prob: float = 1.0
    speaker: str = "S1"


class Transcript(BaseModel):
    language: str = "en"
    duration: float
    model: str
    """e.g. "faster-whisper small (int8, CPU)" or "sidecar"."""
    words: list[Word]
    speakers: list[str] = Field(default_factory=lambda: ["S1"])
    diarization: str = "single"
    """"mfcc-cluster" when speakers were separated by the clustering in recut.asr.diarize, "single" otherwise."""
    diarization_note: str = ""
    seconds: float = 0.0
    """Wall-clock time the transcription took."""


class Sentence(BaseModel):
    index: int
    start: float
    end: float
    text: str
    speaker: str
    first_word: int
    last_word: int
    """Inclusive word indices into Transcript.words."""


class Scores(BaseModel):
    hook: float = 0.0
    """0-10: do the first seconds make a scroller stop?"""
    payoff: float = 0.0
    """0-10: does it deliver a point, a punchline or an answer?"""
    standalone: float = 0.0
    """0-10: understandable without the rest of the video?"""
    clarity: float = 0.0
    """0-10: clean speech, few fillers and false starts."""
    virality: float = 0.0
    """0-100: the model's overall estimate (the number creators sort by)."""


class CutCheck(BaseModel):
    """Deterministic checks of one cut point (recut.clips.snap)."""

    at: float
    inside_word: bool
    gap_before: float
    """Silence between the cut and the nearest word on the left (negative = inside a word)."""
    gap_after: float
    level_db: float | None = None
    """Audio level at the cut relative to the clip's speech level (lower is quieter); None without audio."""


class ClipIssue(BaseModel):
    code: str
    severity: Literal["info", "warning", "error"]
    message: str


class CropKey(BaseModel):
    t: float
    """Seconds from the clip start."""
    cx: float
    """Horizontal centre of the 9:16 window, 0-1 of the source width."""


class CropPlan(BaseModel):
    layout: Literal["track", "fit", "center"]
    keys: list[CropKey] = Field(default_factory=list)
    face_ratio: float = 0.0
    """Share of sampled frames where a face was found."""
    samples: int = 0
    note: str = ""


class RenderInfo(BaseModel):
    path: str
    seconds: float
    """Wall-clock render time."""
    duration: float
    width: int
    height: int
    style: CaptionStyle
    layout: str
    settings_hash: str
    rendered_at: str = Field(default_factory=now_iso)


class ClipPost(BaseModel):
    caption: str = ""
    hashtags: list[str] = Field(default_factory=list)


class Clip(BaseModel):
    id: str
    rank: int = 0
    start: float
    end: float
    first_sentence: int
    last_sentence: int
    title: str
    hook: str = ""
    """The opening line, as spoken."""
    reason: str = ""
    """Why the model picked it."""
    scores: Scores = Field(default_factory=Scores)
    score: float = 0.0
    """Ranking score, 0-100 (recut.clips.suggest.rank_score)."""
    status: ClipStatus = "suggested"
    caption_style: CaptionStyle = "bold"
    layout: Layout = "auto"
    title_card: bool = True
    title_card_text: str = ""
    caption_overrides: dict[int, str] = Field(default_factory=dict)
    """Word index -> corrected text, from the caption editor (the transcript itself is not changed)."""
    cut_in: CutCheck | None = None
    cut_out: CutCheck | None = None
    issues: list[ClipIssue] = Field(default_factory=list)
    adjustments: list[str] = Field(default_factory=list)
    """What the snapping code changed relative to the model's pick ("extended to 21.4 s", ...)."""
    crop: CropPlan | None = None
    render: RenderInfo | None = None
    post: ClipPost = Field(default_factory=ClipPost)
    edited: bool = False
    source: str = "model"
    """"model" (suggested by the LLM), "rules" (the offline baseline) or "manual"."""
    history: list[dict[str, object]] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)


class Chapter(BaseModel):
    start: float
    title: str
    sentence: int = 0


class GroundingIssue(BaseModel):
    kind: Literal["number", "quote", "length", "format"]
    text: str
    message: str


class SocialPost(BaseModel):
    platform: str
    text: str
    limit: int
    chars: int
    within_limit: bool
    truncated: bool = False
    issues: list[GroundingIssue] = Field(default_factory=list)
    approved: bool = False


class Outputs(BaseModel):
    seo_title: str = ""
    seo_description: str = ""
    tags: list[str] = Field(default_factory=list)
    show_notes: str = ""
    blog_post: str = ""
    social: list[SocialPost] = Field(default_factory=list)
    issues: list[GroundingIssue] = Field(default_factory=list)
    """Grounding and length findings for the long-form texts (title, description, notes, blog)."""
    approved: bool = False
    generated_by: str = ""


class SourceInfo(BaseModel):
    path: str
    filename: str
    sha256: str = ""
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_video: bool = True
    size_bytes: int = 0
    title: str = ""
    license: str = ""
    """Licence of the source media, e.g. "Public domain (NASA)", "CC BY 4.0". Required for publishing exports."""
    attribution: str = ""
    """Credit line added to descriptions and posts when the licence needs one."""
    source_url: str = ""


class Stage(BaseModel):
    name: StageName
    state: StageState = "pending"
    seconds: float = 0.0
    message: str = ""


class Project(BaseModel):
    id: str
    name: str
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)
    source: SourceInfo
    stages: list[Stage] = Field(
        default_factory=lambda: [
            Stage(name=n) for n in ("probe", "audio", "transcribe", "diarize", "clips", "chapters", "posts")
        ]
    )
    transcript: Transcript | None = None
    sentences: list[Sentence] = Field(default_factory=list)
    clips: list[Clip] = Field(default_factory=list)
    chapters: list[Chapter] = Field(default_factory=list)
    chapters_text: str = ""
    outputs: Outputs = Field(default_factory=Outputs)
    llm: str = ""
    """The model label that produced clips and texts."""
    error: str = ""
    audit: list[dict[str, object]] = Field(default_factory=list)
    """Review actions (approve, reject, edits, renders, exports) with timestamps."""

    def stage(self, name: StageName) -> Stage:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(name)

    def clip(self, clip_id: str) -> Clip:
        for clip in self.clips:
            if clip.id == clip_id:
                return clip
        raise KeyError(clip_id)

    @property
    def status(self) -> str:
        states = [s.state for s in self.stages]
        if "failed" in states:
            return "failed"
        if "running" in states:
            return "processing"
        if all(s in ("done", "skipped") for s in states):
            return "ready"
        if all(s == "pending" for s in states):
            return "queued"
        return "processing"
