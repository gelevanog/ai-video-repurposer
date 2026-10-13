"use client";

import { Badge, Panel } from "@/components/ui";
import { useApi } from "@/lib/api";
import { pct } from "@/lib/format";

/* eslint-disable @typescript-eslint/no-explicit-any */
type Summary = { runs: Record<string, any>; rubric?: any; all_calls?: any };

function Stat({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone?: string }) {
  return (
    <div className="rounded-2xl border border-line-soft bg-panel p-4">
      <div className="text-[11px] font-medium uppercase tracking-wider text-muted">{label}</div>
      <div className={`mt-1.5 text-[26px] font-semibold ${tone ?? "text-ink"}`}>{value}</div>
      {sub && <div className="mt-0.5 text-[12px] text-muted">{sub}</div>}
    </div>
  );
}

export default function EvaluationPage() {
  const { data } = useApi<Summary>("/api/eval");
  const main = data?.runs?.main;
  const rules = data?.runs?.rules;
  if (!data) return <div className="p-10 text-sm text-muted">Loading…</div>;
  if (!main) return <div className="p-10 text-sm text-muted">No evaluation results yet. Run `make eval-real` (see README).</div>;
  const j = main.judge;
  return (
    <div className="mx-auto max-w-[1280px] px-8 py-8">
      <h1 className="display text-[28px]">Evaluation</h1>
      <p className="mt-1 max-w-3xl text-sm text-muted">
        {main.items} videos, {(main.total_duration_s / 3600).toFixed(1)} hours ({main.synthetic_items} synthetic talks with exact gold data, {main.real_items} public-domain NASA
        briefings with NASA&apos;s captions as reference). Model <span className="mono text-ink-soft">{main.model}</span>, judge{" "}
        <span className="mono text-ink-soft">{main.judge}</span>, speech <span className="mono text-ink-soft">{main.asr}</span>. Finished {main.finished_at}.
      </p>

      <div className="mt-6 grid grid-cols-4 gap-4">
        <Stat label="Caption WER, real speech" value={pct(main.wer.real.wer * 100, 1)} sub={`${main.wer.real.ref_words.toLocaleString()} reference words (NASA captions)`} />
        <Stat label="Caption WER, synthetic" value={pct(main.wer.synthetic.wer * 100, 1)} sub={`${main.wer.synthetic.ref_words.toLocaleString()} words, TTS voices`} />
        <Stat label="Clips judged standalone" value={pct(j.model_clips.standalone_pct)} sub={`vs ${pct(j.random_windows.standalone_pct)} for random windows, n=${j.model_clips.n}`} tone="text-mint" />
        <Stat label="Cuts inside a word" value={`${main.cuts.inside_word} / ${main.cuts.count}`} sub={`first word intact after render: ${pct(main.cuts.first_word_intact_pct)}`} tone="text-mint" />
        <Stat label="Chapter boundaries ±30 s" value={`F1 ${main.chapters.tol30.f1.toFixed(2)}`} sub={rules ? `rules baseline F1 ${rules.chapters.tol30.f1.toFixed(2)}` : undefined} />
        <Stat label="Planted highlights found" value={`${main.highlights.found} / ${main.highlights.planted}`} sub={rules ? `rules baseline ${rules.highlights.found} / ${rules.highlights.planted}` : undefined} />
        <Stat label="Render time" value={`${main.timing.render_seconds_per_output_minute} s`} sub="per minute of 1080×1920 output, CPU" />
        <Stat label="Speech recognition" value={`${main.timing.asr_seconds_per_source_minute} s`} sub="per minute of audio (faster-whisper small, CPU)" />
      </div>

      <div className="mt-6 grid grid-cols-2 gap-5">
        <Panel title="Clip judge (blind, a different free model)">
          <table className="w-full text-sm">
            <thead className="text-left text-[11px] uppercase tracking-wider text-muted">
              <tr>
                <th className="py-1">Group</th>
                <th>n</th>
                <th>Standalone</th>
                <th>Complete ending</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Recut's clips", j.model_clips],
                ["  on synthetic talks", j.model_clips_synthetic],
                ["  on NASA briefings", j.model_clips_real],
                ["Random 25-55 s windows", j.random_windows],
                ["Planted highlights", j.planted_highlights],
              ].map(([label, row]) => (
                <tr key={label as string} className="border-t border-line-soft">
                  <td className="py-1.5 whitespace-pre text-ink-soft">{label as string}</td>
                  <td className="mono">{(row as any).n}</td>
                  <td className="mono">{pct((row as any).standalone_pct)}</td>
                  <td className="mono">{pct((row as any).complete_ending_pct)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
        <Panel title="Model calls">
          <div className="space-y-1.5 text-sm text-ink-soft">
            <div>
              {main.calls.requests} requests ({main.calls.retries_or_errors} retries or errors), {main.calls.input_tokens.toLocaleString()} input and{" "}
              {main.calls.output_tokens.toLocaleString()} output tokens, median latency {main.calls.median_latency_s} s.
            </div>
            <div className="flex flex-wrap gap-1.5">
              {Object.entries(main.calls.served_models).map(([m, n]) => (
                <Badge key={m} tone="mint">
                  {m} × {String(n)}
                </Badge>
              ))}
            </div>
            <div>{main.calls.all_free ? "Every served model was :free (the free-only guard rejects anything else). Cost: $0." : "Some calls were not free."}</div>
          </div>
        </Panel>
      </div>
    </div>
  );
}
