"use client";

import { useSearchParams } from "next/navigation";
import { use, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ClipEditor } from "@/components/ClipEditor";
import { OutputsTab } from "@/components/OutputsTab";
import { type PreviewHandle, PhonePreview } from "@/components/PhonePreview";
import { Timeline } from "@/components/Timeline";
import { TranscriptTab } from "@/components/TranscriptTab";
import { Badge, Button, ScoreRing, StageSteps } from "@/components/ui";
import { api, post, useApi } from "@/lib/api";
import { duration, ts } from "@/lib/format";
import type { AppConfig, Clip, CropPlan, Project } from "@/lib/types";

type Tab = "clips" | "transcript" | "outputs";

export default function ProjectPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const busy = (p: Project | null) => !p || p.status === "processing" || p.status === "queued" || p.jobs.some((j) => j.state === "queued" || j.state === "running");
  const { data: project, setData, reload } = useApi<Project>(`/api/projects/${id}`, { pollMs: 1500, shouldPoll: busy });
  const { data: config } = useApi<AppConfig>("/api/config");
  const audioReady = project?.stages.find((s) => s.name === "audio")?.state === "done";
  const { data: wave } = useApi<{ values: number[] }>(audioReady ? `/api/projects/${id}/waveform?per_second=4` : null);
  const search = useSearchParams();
  const [tab, setTab] = useState<Tab>((search.get("tab") as Tab | null) ?? "clips");
  const [picked, setSelected] = useState<string | null>(search.get("clip"));
  const [mode, setMode] = useState<"live" | "rendered">("live");
  const [playhead, setPlayhead] = useState<number | null>(null);
  const [crop, setCrop] = useState<Record<string, CropPlan>>({});
  const preview = useRef<PreviewHandle | null>(null);

  const selected = picked ?? project?.clips[0]?.id ?? null;
  const clip = useMemo(() => project?.clips.find((c) => c.id === selected) ?? null, [project, selected]);

  // the planned framing (face tracking) for the live preview
  useEffect(() => {
    if (!project || !clip || !project.source.has_video) return;
    const key = `${clip.id}:${clip.start}:${clip.end}:${clip.layout}`;
    if (crop[key] || clip.crop) return;
    api<CropPlan>(`/api/projects/${project.id}/clips/${clip.id}/framing`)
      .then((plan) => setCrop((c) => ({ ...c, [key]: plan })))
      .catch(() => undefined);
  }, [project, clip, crop]);

  const onUpdated = useCallback(
    (updated: Clip) => {
      if (!project) return;
      setData({ ...project, clips: project.clips.map((c) => (c.id === updated.id ? updated : c)) });
      if (!updated.render) setMode("live");
    },
    [project, setData],
  );

  const onTime = useCallback((t: number) => setPlayhead(t), []);

  if (!project) return <div className="p-10 text-sm text-muted">Loading…</div>;
  const words = project.transcript?.words ?? [];
  const renderJob = clip ? [...project.jobs].reverse().find((j) => j.kind === "render" && j.clip_id === clip.id) : undefined;
  const processJob = [...project.jobs].reverse().find((j) => j.kind === "process");
  const cropKey = clip ? `${clip.id}:${clip.start}:${clip.end}:${clip.layout}` : "";
  const approved = project.clips.filter((c) => c.status === "approved").length;

  async function rerun(stage: string) {
    await post(`/api/projects/${id}/process`, { stages: [stage], force: true });
    reload();
  }

  return (
    <div className="px-7 py-6">
      <header className="mb-5 flex items-start justify-between gap-6">
        <div className="min-w-0">
          <div className="flex items-center gap-3">
            <h1 className="display truncate text-[24px]">{project.name}</h1>
            <Badge tone={project.status === "ready" ? "mint" : project.status === "failed" ? "rose" : "violet"}>{project.status}</Badge>
          </div>
          <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[12px] text-muted">
            <span>{duration(project.source.duration)}</span>
            {project.source.has_video ? (
              <span>
                {project.source.width}×{project.source.height} · {project.source.fps} fps
              </span>
            ) : (
              <span>audio only</span>
            )}
            {project.transcript && <span>{project.transcript.speakers.length === 1 ? "1 speaker" : `${project.transcript.speakers.length} speakers`}</span>}
            <span>{project.source.license || <span className="text-amber">licence not set</span>}</span>
            {project.llm && <span className="mono">{project.llm}</span>}
          </div>
          <div className="mt-3">
            <StageSteps stages={project.stages} />
          </div>
          {project.error && <p className="mt-2 text-xs text-rose">{project.error}</p>}
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Button variant="ghost" onClick={() => rerun("clips")} disabled={busy(project)}>
            ↻ Find clips again
          </Button>
          <div className="rounded-xl border border-line-soft bg-panel px-3 py-1.5 text-right text-[11px] text-muted">
            <div className="text-lg font-semibold text-ink">
              {approved}/{project.clips.length}
            </div>
            clips approved
          </div>
        </div>
      </header>

      <nav className="mb-5 flex gap-1 border-b border-line-soft">
        {(["clips", "transcript", "outputs"] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`-mb-px border-b-2 px-3.5 py-2 text-sm ${tab === t ? "border-violet text-ink" : "border-transparent text-muted hover:text-ink-soft"}`}
          >
            {t === "clips" ? `Clips (${project.clips.length})` : t === "transcript" ? "Transcript" : "Chapters & posts"}
          </button>
        ))}
      </nav>

      {tab === "clips" && (
        <div className="space-y-5">
          <Timeline
            duration={project.source.duration}
            wave={wave?.values ?? []}
            sentences={project.sentences}
            chapters={project.chapters}
            clips={project.clips}
            selected={selected}
            playhead={playhead}
            onSelect={(cid) => {
              setSelected(cid);
              setMode("live");
            }}
          />
          {!project.clips.length && <p className="text-sm text-muted">{busy(project) ? `Working: ${processJob?.state ?? "queued"}…` : "No clips found."}</p>}
          {clip && (
            <div className="grid grid-cols-[290px_340px_minmax(0,1fr)] gap-5">
              <div className="scroll-thin max-h-[860px] space-y-2 overflow-y-auto pr-1">
                {project.clips.map((c) => (
                  <button
                    key={c.id}
                    onClick={() => {
                      setSelected(c.id);
                      setMode("live");
                    }}
                    className={`w-full rounded-2xl border p-3 text-left transition ${c.id === selected ? "border-violet bg-violet-soft/60" : "border-line-soft bg-panel hover:border-line"} ${c.status === "rejected" ? "opacity-50" : ""}`}
                  >
                    <div className="flex items-start gap-2.5">
                      <ScoreRing score={c.score} size={38} />
                      <div className="min-w-0 flex-1">
                        <div className="line-clamp-2 text-[13px] font-semibold leading-snug">{c.title}</div>
                        <div className="mono mt-1 text-[10.5px] text-muted">
                          #{c.rank} · {ts(c.start)} · {Math.round(c.end - c.start)} s
                        </div>
                      </div>
                    </div>
                    <p className="mt-2 line-clamp-2 text-[11.5px] italic leading-snug text-muted">“{c.hook}”</p>
                    <div className="mt-2 flex flex-wrap gap-1">
                      {c.status === "approved" && <Badge tone="lime">approved</Badge>}
                      {c.render && <Badge tone="sky">rendered</Badge>}
                      {c.issues.some((i) => i.severity === "error") && <Badge tone="rose">needs a fix</Badge>}
                      {c.issues.filter((i) => i.severity === "warning").length > 0 && <Badge tone="amber">{c.issues.filter((i) => i.severity === "warning").length} warnings</Badge>}
                      {c.issues.length === 0 && <Badge tone="mint">clean cuts</Badge>}
                    </div>
                  </button>
                ))}
              </div>
              <div className="flex flex-col items-center">
                <div className="mb-3 flex rounded-lg border border-line-soft bg-panel p-0.5 text-[12px]">
                  {(["live", "rendered"] as const).map((m) => (
                    <button
                      key={m}
                      disabled={m === "rendered" && !clip.render}
                      onClick={() => setMode(m)}
                      className={`rounded-md px-3 py-1 disabled:opacity-40 ${mode === m ? "bg-panel-3 text-ink" : "text-muted"}`}
                    >
                      {m === "live" ? "Live preview" : "Rendered MP4"}
                    </button>
                  ))}
                </div>
                <PhonePreview
                  key={`${clip.id}-${mode}`}
                  projectId={project.id}
                  clip={clip}
                  words={words}
                  source={project.source}
                  crop={crop[cropKey] ?? clip.crop}
                  mode={mode}
                  onTime={onTime}
                  handleRef={preview}
                />
                {(crop[cropKey] ?? clip.crop)?.note && <p className="mt-2 w-[300px] text-center text-[11px] text-muted">{(crop[cropKey] ?? clip.crop)?.note}</p>}
              </div>
              <div className="rounded-2xl border border-line-soft bg-panel p-4">
                <ClipEditor project={project} clip={clip} onUpdated={onUpdated} onJob={() => reload()} renderJob={renderJob} />
              </div>
            </div>
          )}
        </div>
      )}

      {tab === "transcript" && (
        <TranscriptTab
          project={project}
          onPick={(cid) => {
            setSelected(cid);
            setTab("clips");
          }}
        />
      )}
      {tab === "outputs" && <OutputsTab project={project} config={config} onChange={(p) => setData({ ...project, ...p })} />}
    </div>
  );
}
