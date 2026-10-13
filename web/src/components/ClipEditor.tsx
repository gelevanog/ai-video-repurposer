"use client";

import { useMemo, useRef, useState } from "react";

import { Badge, Button, Meter, ScoreRing } from "@/components/ui";
import { API_URL, patch, post } from "@/lib/api";
import { STYLES } from "@/lib/captions";
import { duration as fmtDuration, ts } from "@/lib/format";
import type { CaptionStyle, Clip, Job, Layout, Project } from "@/lib/types";

const STYLE_ORDER: CaptionStyle[] = ["bold", "boxed", "karaoke", "pop", "minimal"];
const STYLE_NAMES: Record<CaptionStyle, string> = { bold: "Bold", boxed: "Boxed", karaoke: "Karaoke", pop: "Pop", minimal: "Minimal" };

export function ClipEditor({
  project,
  clip,
  onUpdated,
  onJob,
  renderJob,
}: {
  project: Project;
  clip: Clip;
  onUpdated: (clip: Clip) => void;
  onJob: (job: Job) => void;
  renderJob: Job | undefined;
}) {
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  async function save(body: Record<string, unknown>) {
    setSaving(true);
    setError(null);
    try {
      onUpdated(await patch<Clip>(`/api/projects/${project.id}/clips/${clip.id}`, body));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }

  async function render() {
    setError(null);
    try {
      onJob(await post<Job>(`/api/projects/${project.id}/clips/${clip.id}/render`));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  const rendering = renderJob && (renderJob.state === "queued" || renderJob.state === "running");
  const errors = clip.issues.filter((i) => i.severity === "error");

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3">
        <ScoreRing score={clip.score} size={50} />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="mono text-xs text-muted">#{clip.rank}</span>
            <input
              key={clip.id + clip.title}
              defaultValue={clip.title}
              onBlur={(e) => e.target.value !== clip.title && save({ title: e.target.value })}
              className="min-w-0 flex-1 rounded-md border border-transparent bg-transparent px-1 py-0.5 text-[15px] font-semibold hover:border-line focus:border-violet focus:outline-none"
            />
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[11px]">
            <Badge tone={clip.status === "approved" ? "lime" : clip.status === "rejected" ? "rose" : "violet"}>{clip.status}</Badge>
            <Badge>
              {ts(clip.start, true)} → {ts(clip.end, true)}
            </Badge>
            <Badge tone={clip.end - clip.start > 60 || clip.end - clip.start < 20 ? "amber" : "neutral"}>{fmtDuration(clip.end - clip.start)}</Badge>
            {clip.source === "manual" && <Badge tone="sky">added by you</Badge>}
            {clip.edited && <Badge tone="sky">edited</Badge>}
          </div>
        </div>
      </div>

      {clip.reason && (
        <p className="rounded-xl border border-line-soft bg-panel-2 px-3 py-2 text-[13px] leading-relaxed text-ink-soft">
          <span className="mr-1 text-[11px] font-semibold uppercase tracking-wider text-violet">Why this clip</span> {clip.reason}
        </p>
      )}

      <div className="grid grid-cols-2 gap-x-5 gap-y-1.5">
        <Meter label="Hook" value={clip.scores.hook} />
        <Meter label="Payoff" value={clip.scores.payoff} />
        <Meter label="Standalone" value={clip.scores.standalone} />
        <Meter label="Clarity" value={clip.scores.clarity} />
        <Meter label="Virality" value={clip.scores.virality} max={100} />
      </div>

      <TrimBar project={project} clip={clip} onTrim={(body) => save(body)} />

      <CutChecks clip={clip} />

      {clip.issues.length > 0 && (
        <ul className="space-y-1">
          {clip.issues.map((issue) => (
            <li key={issue.code + issue.message} className="flex items-start gap-2 text-[12px]">
              <span className={`mt-1 h-2 w-2 shrink-0 rounded-full ${issue.severity === "error" ? "bg-rose" : issue.severity === "warning" ? "bg-amber" : "bg-sky"}`} />
              <span className="text-ink-soft">{issue.message}</span>
            </li>
          ))}
        </ul>
      )}

      <div>
        <h4 className="mb-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted">Caption style</h4>
        <div className="grid grid-cols-5 gap-1.5">
          {STYLE_ORDER.map((style) => {
            const spec = STYLES[style];
            const on = clip.caption_style === style;
            return (
              <button
                key={style}
                onClick={() => save({ caption_style: style })}
                className={`flex h-[62px] flex-col items-center justify-center rounded-xl border text-[10px] ${on ? "border-violet bg-violet-soft" : "border-line-soft bg-panel-2 hover:border-muted"}`}
              >
                <span
                  className="text-[17px] leading-none"
                  style={{
                    fontFamily: spec.font,
                    color: spec.pop || spec.karaoke ? spec.highlight : spec.color,
                    background: spec.box ?? undefined,
                    padding: spec.box ? "2px 4px" : undefined,
                    borderRadius: 3,
                    WebkitTextStroke: spec.box ? undefined : `1px ${spec.outline}`,
                  }}
                >
                  {spec.uppercase ? "AA" : "Aa"}
                </span>
                <span className="mt-1.5 text-muted">{STYLE_NAMES[style]}</span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <label className="text-[11px] text-muted">
          Framing
          <select
            value={clip.layout}
            onChange={(e) => save({ layout: e.target.value as Layout })}
            className="mt-1 w-full rounded-lg border border-line bg-panel-2 px-2 py-1.5 text-sm text-ink"
          >
            <option value="auto">Auto (follow faces, else full frame)</option>
            <option value="track">Follow the speaker&apos;s face</option>
            <option value="fit">Full frame, blurred fill</option>
            <option value="center">Centre crop</option>
          </select>
        </label>
        <label className="text-[11px] text-muted">
          <span className="flex items-center justify-between">
            Title card
            <input type="checkbox" checked={clip.title_card} onChange={(e) => save({ title_card: e.target.checked })} className="accent-violet" />
          </span>
          <input
            key={clip.id + clip.title_card_text}
            defaultValue={clip.title_card_text}
            disabled={!clip.title_card}
            onBlur={(e) => e.target.value !== clip.title_card_text && save({ title_card_text: e.target.value })}
            className="mt-1 w-full rounded-lg border border-line bg-panel-2 px-2 py-1.5 text-sm text-ink disabled:opacity-40"
          />
        </label>
      </div>

      <CaptionWords project={project} clip={clip} onSave={(overrides) => save({ caption_overrides: overrides })} />

      <div className="flex flex-wrap items-center gap-2 border-t border-line-soft pt-4">
        {clip.status !== "approved" ? (
          <Button variant="success" onClick={() => save({ status: "approved" })} disabled={saving} title={errors.length ? "Fix the errors first" : "Approve for download and export"}>
            ✓ Approve
          </Button>
        ) : (
          <Button onClick={() => save({ status: "suggested" })}>Unapprove</Button>
        )}
        {clip.status !== "rejected" && (
          <Button variant="ghost" onClick={() => save({ status: "rejected" })}>
            Reject
          </Button>
        )}
        <Button variant="primary" onClick={render} disabled={!!rendering}>
          {rendering ? "Rendering…" : clip.render ? "Re-render" : "Render MP4"}
        </Button>
        {clip.render && clip.status === "approved" ? (
          <a href={`${API_URL}/api/projects/${project.id}/clips/${clip.id}/video?download=1`} className="rounded-lg border border-line bg-panel-3 px-3 py-1.5 text-sm hover:border-muted">
            ↓ Download
          </a>
        ) : (
          <span className="text-[11px] text-muted">{clip.render ? "Approve to download" : "Render, then approve to download"}</span>
        )}
      </div>
      {clip.render && (
        <p className="text-[11px] text-muted">
          Rendered {clip.render.width}×{clip.render.height}, {clip.render.layout}, in {clip.render.seconds.toFixed(1)} s on CPU ({((clip.render.seconds / clip.render.duration) * 60).toFixed(0)} s per minute of clip).
        </p>
      )}
      {renderJob?.state === "failed" && <p className="text-xs text-rose">Render failed: {renderJob.error}</p>}
      {error && <p className="rounded-lg border border-rose/30 bg-rose-soft px-3 py-2 text-xs text-rose">{error}</p>}
    </div>
  );
}

function CutChecks({ clip }: { clip: Clip }) {
  const rows = [
    ["In", clip.cut_in],
    ["Out", clip.cut_out],
  ] as const;
  return (
    <div className="grid grid-cols-2 gap-2">
      {rows.map(([label, cut]) =>
        cut ? (
          <div key={label} className="rounded-xl border border-line-soft bg-panel-2 px-3 py-2 text-[11px]">
            <div className="flex items-center justify-between">
              <span className="font-semibold text-ink-soft">Cut {label.toLowerCase()}</span>
              {cut.inside_word ? <Badge tone="rose">mid-word</Badge> : <Badge tone="mint">clean</Badge>}
            </div>
            <div className="mono mt-1 text-muted">
              {ts(cut.at, true)} · gap {Math.round(Math.max(0, label === "In" ? cut.gap_after : cut.gap_before) * 1000)} ms
              {cut.level_db !== null ? ` · ${cut.level_db.toFixed(0)} dB` : ""}
            </div>
          </div>
        ) : null,
      )}
    </div>
  );
}

function TrimBar({ project, clip, onTrim }: { project: Project; clip: Clip; onTrim: (body: Record<string, unknown>) => void }) {
  const words = useMemo(() => project.transcript?.words ?? [], [project.transcript]);
  const pad = 6;
  const ws = Math.max(0, clip.start - pad);
  const we = Math.min(project.source.duration, clip.end + pad);
  const bar = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<{ which: "start" | "end"; start: number; end: number } | null>(null);
  const range = drag ? { start: drag.start, end: drag.end } : { start: clip.start, end: clip.end };
  const x = (t: number) => `${((t - ws) / (we - ws)) * 100}%`;
  const visible = words.filter((w) => w.end > ws && w.start < we);
  const sentences = project.sentences;

  function timeAt(clientX: number): number {
    const rect = bar.current?.getBoundingClientRect();
    if (!rect) return 0;
    return ws + ((clientX - rect.left) / rect.width) * (we - ws);
  }

  function onMove(e: React.PointerEvent) {
    if (!drag) return;
    const t = timeAt(e.clientX);
    setDrag(drag.which === "start" ? { ...drag, start: Math.min(t, drag.end - 3) } : { ...drag, end: Math.max(t, drag.start + 3) });
  }

  function onUp() {
    if (!drag) return;
    setDrag(null);
    if (drag.start !== clip.start || drag.end !== clip.end) onTrim({ start: drag.start, end: drag.end, snap: "word" });
  }

  const first = sentences[clip.first_sentence];
  const last = sentences[clip.last_sentence];
  const prev = sentences[clip.first_sentence - 1];
  const next = sentences[clip.last_sentence + 1];

  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between">
        <h4 className="text-[11px] font-semibold uppercase tracking-wider text-muted">Trim · cuts snap to the silence between words</h4>
        <span className="mono text-[11px] text-muted">{ts(range.end - range.start, true)}</span>
      </div>
      <div ref={bar} className="relative h-[54px] touch-none rounded-xl border border-line-soft bg-panel-2" onPointerMove={onMove} onPointerUp={onUp} onPointerLeave={onUp}>
        <div className="absolute inset-y-0 rounded-lg bg-violet/15" style={{ left: x(range.start), width: `calc(${x(range.end)} - ${x(range.start)})` }} />
        {visible.map((w, i) => {
          const inside = w.start >= range.start - 0.01 && w.end <= range.end + 0.01;
          return (
            <div
              key={i}
              title={w.text}
              className="absolute top-[18px] h-[18px] rounded-[3px]"
              style={{ left: x(w.start), width: `calc(${x(w.end)} - ${x(w.start)})`, background: inside ? (w.prob < 0.5 ? "#fbbf24" : "#a78bfa") : "#3a4058", opacity: inside ? 0.9 : 0.6 }}
            />
          );
        })}
        {(["start", "end"] as const).map((which) => (
          <div
            key={which}
            onPointerDown={(e) => {
              (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
              setDrag({ which, start: clip.start, end: clip.end });
            }}
            className="absolute inset-y-[-4px] z-10 w-3 -translate-x-1/2 cursor-ew-resize rounded-md bg-white shadow"
            style={{ left: x(range[which]) }}
            title={which === "start" ? "Drag the cut-in" : "Drag the cut-out"}
          >
            <span className="absolute inset-y-3 left-1/2 w-px -translate-x-1/2 bg-panel-3" />
          </div>
        ))}
      </div>
      <div className="mt-2 flex flex-wrap gap-1.5">
        <Button variant="ghost" className="!px-2 !py-1 text-[12px]" disabled={!prev} onClick={() => prev && onTrim({ start: prev.start, end: clip.end, snap: "sentence" })}>
          ← Start a sentence earlier
        </Button>
        <Button variant="ghost" className="!px-2 !py-1 text-[12px]" disabled={first?.index === last?.index} onClick={() => sentences[clip.first_sentence + 1] && onTrim({ start: sentences[clip.first_sentence + 1].start, end: clip.end, snap: "sentence" })}>
          Start later →
        </Button>
        <Button variant="ghost" className="!px-2 !py-1 text-[12px]" disabled={first?.index === last?.index} onClick={() => sentences[clip.last_sentence - 1] && onTrim({ start: clip.start, end: sentences[clip.last_sentence - 1].end, snap: "sentence" })}>
          ← End earlier
        </Button>
        <Button variant="ghost" className="!px-2 !py-1 text-[12px]" disabled={!next} onClick={() => next && onTrim({ start: clip.start, end: next.end, snap: "sentence" })}>
          End a sentence later →
        </Button>
      </div>
    </div>
  );
}

function CaptionWords({ project, clip, onSave }: { project: Project; clip: Clip; onSave: (overrides: Record<string, string>) => void }) {
  const [editing, setEditing] = useState<number | null>(null);
  const words = project.transcript?.words ?? [];
  const indices: number[] = [];
  words.forEach((w, i) => {
    if (w.end > clip.start && w.start < clip.end) indices.push(i);
  });
  const unsure = indices.filter((i) => words[i].prob < 0.5).length;
  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between">
        <h4 className="text-[11px] font-semibold uppercase tracking-wider text-muted">Captions · click a word to fix it</h4>
        {unsure > 0 && <span className="text-[11px] text-amber">{unsure} low-confidence words</span>}
      </div>
      <div className="scroll-thin max-h-[150px] overflow-y-auto rounded-xl border border-line-soft bg-panel-2 p-2.5 text-[13px] leading-7">
        {indices.map((i) => {
          const override = clip.caption_overrides[String(i)];
          const text = override ?? words[i].text;
          if (editing === i)
            return (
              <input
                key={i}
                autoFocus
                defaultValue={text}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    onSave({ [i]: (e.target as HTMLInputElement).value });
                    setEditing(null);
                  }
                  if (e.key === "Escape") setEditing(null);
                }}
                onBlur={(e) => {
                  if (e.target.value !== text) onSave({ [i]: e.target.value });
                  setEditing(null);
                }}
                className="mx-0.5 w-24 rounded border border-violet bg-panel px-1 text-ink focus:outline-none"
              />
            );
          return (
            <button
              key={i}
              onClick={() => setEditing(i)}
              title={override !== undefined ? `was “${words[i].text}”` : `confidence ${(words[i].prob * 100).toFixed(0)}%`}
              className={`mx-[1px] rounded px-[3px] hover:bg-panel-3 ${override !== undefined ? "bg-sky-soft text-sky" : words[i].prob < 0.5 ? "text-amber underline decoration-dotted" : "text-ink-soft"} ${override === "" ? "line-through opacity-50" : ""}`}
            >
              {override === "" ? words[i].text : text}
            </button>
          );
        })}
      </div>
    </div>
  );
}
