"use client";

import { useEffect, useRef } from "react";

import { scoreColor, speakerColor, ts } from "@/lib/format";
import type { Chapter, Clip, Sentence } from "@/lib/types";

/** The whole video: loudness, speaker turns, chapters and the suggested clips; click a clip to edit it. */
export function Timeline({
  duration,
  wave,
  sentences,
  chapters,
  clips,
  selected,
  playhead,
  onSelect,
}: {
  duration: number;
  wave: number[];
  sentences: Sentence[];
  chapters: Chapter[];
  clips: Clip[];
  selected: string | null;
  playhead: number | null;
  onSelect: (id: string) => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const el = canvas.current;
    if (!el || !wave.length) return;
    const dpr = window.devicePixelRatio || 1;
    const width = el.clientWidth;
    const height = el.clientHeight;
    el.width = width * dpr;
    el.height = height * dpr;
    const ctx = el.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);
    const bars = Math.min(wave.length, Math.floor(width / 2));
    const per = wave.length / bars;
    const step = width / bars;
    ctx.fillStyle = "#3a4058";
    for (let i = 0; i < bars; i++) {
      let peak = 0;
      for (let j = Math.floor(i * per); j < Math.max(Math.floor((i + 1) * per), Math.floor(i * per) + 1); j++) peak = Math.max(peak, wave[j] ?? 0);
      const h = Math.max(1, peak * (height - 4));
      ctx.fillRect(i * step, (height - h) / 2, Math.max(1, step * 0.6), h);
    }
  }, [wave]);

  const ticks = Math.max(1, Math.round(duration / 8 / 60)) * 60;
  const marks: number[] = [];
  for (let s = 0; s <= duration; s += ticks) marks.push(s);
  const pct = (s: number) => `${(s / Math.max(1, duration)) * 100}%`;

  return (
    <div className="select-none">
      <div className="relative h-5 text-[10px] text-muted">
        {marks.map((m) => (
          <span key={m} className="mono absolute -translate-x-1/2" style={{ left: pct(m) }}>
            {ts(m)}
          </span>
        ))}
      </div>
      <div className="relative h-[118px] overflow-hidden rounded-xl border border-line-soft bg-panel-2">
        {/* speaker turns */}
        <div className="absolute inset-x-0 top-0 h-1.5">
          {sentences.map((s) => (
            <div key={s.index} className="absolute top-0 h-full" style={{ left: pct(s.start), width: pct(s.end - s.start), background: speakerColor(s.speaker), opacity: 0.7 }} />
          ))}
        </div>
        <canvas ref={canvas} className="absolute inset-x-0 top-3 h-[52px] w-full" />
        {/* chapters */}
        {chapters.map((c, i) => {
          const next = chapters[i + 1]?.start ?? duration;
          return (
            <div key={c.start} className="absolute top-1.5 h-[62px] border-l border-dashed border-violet/60" style={{ left: pct(c.start), width: pct(next - c.start) }}>
              <span className="absolute left-1 right-1 top-0 truncate rounded bg-panel-2/90 px-1 text-[10px] text-violet-200" title={c.title}>
                {c.title}
              </span>
            </div>
          );
        })}
        {/* clips */}
        <div className="absolute inset-x-0 bottom-2 h-[40px]">
          {clips.map((clip) => {
            const isSel = clip.id === selected;
            const color = clip.status === "rejected" ? "#4b5163" : scoreColor(clip.score);
            return (
              <button
                key={clip.id}
                onClick={() => onSelect(clip.id)}
                title={`#${clip.rank} ${clip.title} (${ts(clip.start)}-${ts(clip.end)}, score ${Math.round(clip.score)})`}
                className={`absolute top-0 h-full overflow-hidden rounded-md border text-left text-[10px] font-semibold transition ${isSel ? "z-10 ring-2 ring-white" : "hover:brightness-125"}`}
                style={{ left: pct(clip.start), width: `max(${pct(clip.end - clip.start)}, 22px)`, background: `${color}33`, borderColor: color, color }}
              >
                <span className="px-1">
                  {clip.status === "approved" ? "✓ " : ""}#{clip.rank}
                </span>
              </button>
            );
          })}
        </div>
        {playhead !== null && <div className="pointer-events-none absolute inset-y-0 z-20 w-px bg-white" style={{ left: pct(playhead) }} />}
      </div>
    </div>
  );
}
