"use client";

import { Badge } from "@/components/ui";
import { speakerColor, ts } from "@/lib/format";
import type { Project, Sentence } from "@/lib/types";

/** Speaker turns with timestamps, chapters inline and the clips marked in the margin. */
export function TranscriptTab({ project, onPick }: { project: Project; onPick: (clipId: string) => void }) {
  const t = project.transcript;
  if (!t) return <p className="text-sm text-muted">No transcript yet.</p>;
  const turns: Sentence[][] = [];
  for (const s of project.sentences) {
    const last = turns[turns.length - 1];
    if (last && last[0].speaker === s.speaker && s.start - last[last.length - 1].end < 2.5 && last.length < 8) last.push(s);
    else turns.push([s]);
  }
  const chapterAt = new Map(project.chapters.map((c) => [c.sentence, c]));
  const clipAt = new Map<number, string>();
  for (const c of project.clips) clipAt.set(c.first_sentence, c.id);
  const lowConfidence = t.words.filter((w) => w.prob < 0.5).length;

  return (
    <div className="grid grid-cols-[1fr_260px] gap-6">
      <div className="space-y-3">
        {turns.map((turn) => {
          const first = turn[0];
          const chapter = turn.map((s) => chapterAt.get(s.index)).find(Boolean);
          const clipId = turn.map((s) => clipAt.get(s.index)).find(Boolean);
          return (
            <div key={first.index}>
              {chapter && (
                <h3 className="mb-2 mt-5 flex items-center gap-2 text-sm font-semibold text-violet-200">
                  <span className="mono text-[11px] text-violet">{ts(chapter.start)}</span> {chapter.title}
                </h3>
              )}
              <div className="flex gap-3">
                <div className="w-[64px] shrink-0 pt-0.5 text-right">
                  <div className="text-[11px] font-semibold" style={{ color: speakerColor(first.speaker) }}>
                    {first.speaker}
                  </div>
                  <div className="mono text-[10px] text-muted">{ts(first.start)}</div>
                </div>
                <p className="flex-1 text-[13.5px] leading-relaxed text-ink-soft">
                  {turn.map((s) => s.text).join(" ")}
                  {clipId && (
                    <button onClick={() => onPick(clipId)} className="ml-2 align-middle">
                      <Badge tone="lime">clip starts here</Badge>
                    </button>
                  )}
                </p>
              </div>
            </div>
          );
        })}
      </div>
      <aside className="space-y-3 text-[12px] text-muted">
        <div className="rounded-2xl border border-line-soft bg-panel p-3.5">
          <div className="mb-1 font-semibold text-ink-soft">Speech recognition</div>
          <div>{t.model}</div>
          <div>
            {t.words.length.toLocaleString()} words in {t.seconds.toFixed(0)} s ({((t.seconds / Math.max(1, t.duration)) * 60).toFixed(1)} s per minute of audio)
          </div>
          {lowConfidence > 0 && <div className="text-amber">{lowConfidence} low-confidence words (check names and numbers)</div>}
        </div>
        <div className="rounded-2xl border border-line-soft bg-panel p-3.5">
          <div className="mb-1 font-semibold text-ink-soft">Speakers</div>
          <div className="mb-1.5 flex gap-1.5">
            {t.speakers.map((s) => (
              <span key={s} className="rounded px-1.5 py-0.5 text-[11px] font-semibold" style={{ background: `${speakerColor(s)}22`, color: speakerColor(s) }}>
                {s}
              </span>
            ))}
          </div>
          <div>{t.diarization_note || "Single speaker."}</div>
          <div className="mt-1">Light CPU clustering (MFCC + pitch), not a neural diarizer: check speaker labels on overlapping speech.</div>
        </div>
      </aside>
    </div>
  );
}
