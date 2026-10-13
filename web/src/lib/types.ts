export type StageName = "probe" | "audio" | "transcribe" | "diarize" | "clips" | "chapters" | "posts";
export type StageState = "pending" | "running" | "done" | "failed" | "skipped";
export type CaptionStyle = "bold" | "boxed" | "karaoke" | "pop" | "minimal";
export type Layout = "auto" | "track" | "fit" | "center";
export type ClipStatus = "suggested" | "approved" | "rejected";

export interface Stage {
  name: StageName;
  state: StageState;
  seconds: number;
  message: string;
}

export interface Word {
  start: number;
  end: number;
  text: string;
  prob: number;
  speaker: string;
}

export interface Transcript {
  language: string;
  duration: number;
  model: string;
  words: Word[];
  speakers: string[];
  diarization: string;
  diarization_note: string;
  seconds: number;
}

export interface Sentence {
  index: number;
  start: number;
  end: number;
  text: string;
  speaker: string;
  first_word: number;
  last_word: number;
}

export interface Scores {
  hook: number;
  payoff: number;
  standalone: number;
  clarity: number;
  virality: number;
}

export interface CutCheck {
  at: number;
  inside_word: boolean;
  gap_before: number;
  gap_after: number;
  level_db: number | null;
}

export interface ClipIssue {
  code: string;
  severity: "info" | "warning" | "error";
  message: string;
}

export interface CropKey {
  t: number;
  cx: number;
}

export interface CropPlan {
  layout: "track" | "fit" | "center";
  keys: CropKey[];
  face_ratio: number;
  samples: number;
  note: string;
}

export interface RenderInfo {
  path: string;
  seconds: number;
  duration: number;
  width: number;
  height: number;
  style: CaptionStyle;
  layout: string;
  settings_hash: string;
  rendered_at: string;
}

export interface Clip {
  id: string;
  rank: number;
  start: number;
  end: number;
  first_sentence: number;
  last_sentence: number;
  title: string;
  hook: string;
  reason: string;
  scores: Scores;
  score: number;
  status: ClipStatus;
  caption_style: CaptionStyle;
  layout: Layout;
  title_card: boolean;
  title_card_text: string;
  caption_overrides: Record<string, string>;
  cut_in: CutCheck | null;
  cut_out: CutCheck | null;
  issues: ClipIssue[];
  adjustments: string[];
  crop: CropPlan | null;
  render: RenderInfo | null;
  post: { caption: string; hashtags: string[] };
  edited: boolean;
  source: string;
}

export interface Chapter {
  start: number;
  title: string;
  sentence: number;
}

export interface GroundingIssue {
  kind: "number" | "quote" | "length" | "format";
  text: string;
  message: string;
}

export interface SocialPost {
  platform: string;
  text: string;
  limit: number;
  chars: number;
  within_limit: boolean;
  truncated: boolean;
  issues: GroundingIssue[];
  approved: boolean;
}

export interface Outputs {
  seo_title: string;
  seo_description: string;
  tags: string[];
  show_notes: string;
  blog_post: string;
  social: SocialPost[];
  issues: GroundingIssue[];
  approved: boolean;
  generated_by: string;
}

export interface SourceInfo {
  path: string;
  filename: string;
  sha256: string;
  duration: number;
  width: number;
  height: number;
  fps: number;
  has_video: boolean;
  size_bytes: number;
  title: string;
  license: string;
  attribution: string;
  source_url: string;
}

export interface Job {
  id: string;
  kind: "process" | "render";
  project_id: string;
  clip_id: string;
  state: "queued" | "running" | "done" | "failed";
  error: string;
  seconds: number;
}

export interface Project {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
  status: string;
  source: SourceInfo;
  stages: Stage[];
  transcript: Transcript | null;
  sentences: Sentence[];
  clips: Clip[];
  chapters: Chapter[];
  chapters_text: string;
  outputs: Outputs;
  llm: string;
  error: string;
  audit: Record<string, unknown>[];
  jobs: Job[];
}

export interface ProjectSummary {
  id: string;
  name: string;
  status: string;
  created_at: string;
  duration: number;
  has_video: boolean;
  license: string;
  clips: number;
  approved: number;
  rendered: number;
  speakers: number;
  chapters: number;
  llm: string;
  error: string;
  stages: Stage[];
}

export interface AppConfig {
  version: string;
  llm_provider: string;
  llm_model: string;
  free_only: boolean;
  asr: string;
  whisper_model: string;
  clip_min_seconds: number;
  clip_max_seconds: number;
  styles: Record<CaptionStyle, string>;
  platforms: Record<string, { name: string; limit: number; target: number }>;
  youtube: { title: number; title_target: number; description: number };
  samples: { name: string; path: string; license: string; attribution: string }[];
}
