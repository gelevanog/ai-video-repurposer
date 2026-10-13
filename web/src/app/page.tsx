"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { Badge, Button, Panel, StageSteps } from "@/components/ui";
import { API_URL, api, post, useApi } from "@/lib/api";
import { duration } from "@/lib/format";
import type { AppConfig, ProjectSummary } from "@/lib/types";

const LICENSES = ["Public domain", "CC0 1.0", "CC BY 4.0", "CC BY-SA 4.0", "My own content", "Licensed from the creator (client work)"];

export default function ProjectsPage() {
  const router = useRouter();
  const { data: projects, reload } = useApi<ProjectSummary[]>("/api/projects", {
    pollMs: 2500,
    shouldPoll: (d) => !!d?.some((p) => p.status === "processing" || p.status === "queued"),
  });
  const { data: config } = useApi<AppConfig>("/api/config");
  const [file, setFile] = useState<File | null>(null);
  const [license, setLicense] = useState(LICENSES[0]);
  const [attribution, setAttribution] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  async function upload() {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("license", license);
      form.append("attribution", attribution);
      const created = await api<{ project: ProjectSummary }>("/api/projects", { method: "POST", body: form });
      router.push(`/p/${created.project.id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function importSample(sample: AppConfig["samples"][number]) {
    setBusy(true);
    try {
      const created = await post<{ project: ProjectSummary }>("/api/projects/from-path", {
        path: sample.path,
        name: sample.name,
        license: sample.license,
        attribution: sample.attribution,
      });
      router.push(`/p/${created.project.id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      reload();
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto max-w-[1280px] px-8 py-8">
      <header className="mb-7 flex items-end justify-between gap-6">
        <div>
          <h1 className="display text-[28px]">Projects</h1>
          <p className="mt-1 max-w-2xl text-sm text-muted">
            Drop in a long video or podcast. Recut transcribes it on this machine, finds the moments that stand on their own, frames them
            vertically with word-by-word captions, and drafts chapters, show notes, a blog post and social posts. You review and approve
            everything before it leaves.
          </p>
        </div>
      </header>

      <div className="grid grid-cols-[1fr_380px] gap-6">
        <div className="space-y-3">
          {!projects && <div className="text-sm text-muted">Loading…</div>}
          {projects?.length === 0 && <div className="rounded-2xl border border-dashed border-line p-10 text-center text-muted">No projects yet.</div>}
          {projects?.map((p) => (
            <Link
              key={p.id}
              href={`/p/${p.id}`}
              className="flex gap-4 rounded-2xl border border-line-soft bg-panel p-3 transition hover:border-violet/50 hover:bg-panel-2"
            >
              <div className="relative h-[92px] w-[164px] shrink-0 overflow-hidden rounded-xl bg-panel-3">
                {p.has_video && p.status !== "queued" && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={`${API_URL}/api/projects/${p.id}/thumb`} alt="" className="h-full w-full object-cover" onError={(e) => (e.currentTarget.style.display = "none")} />
                )}
                <span className="mono absolute bottom-1.5 right-1.5 rounded bg-black/70 px-1.5 py-0.5 text-[10px] text-white">{duration(p.duration)}</span>
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-start justify-between gap-3">
                  <h3 className="truncate font-semibold">{p.name}</h3>
                  <Badge tone={p.status === "ready" ? "mint" : p.status === "failed" ? "rose" : "violet"}>{p.status}</Badge>
                </div>
                <div className="mt-1 flex flex-wrap gap-1.5 text-[11px]">
                  <Badge>{p.clips} clips</Badge>
                  {p.approved > 0 && <Badge tone="lime">{p.approved} approved</Badge>}
                  {p.rendered > 0 && <Badge tone="sky">{p.rendered} rendered</Badge>}
                  {p.speakers > 0 && <Badge>{p.speakers === 1 ? "1 speaker" : `${p.speakers} speakers`}</Badge>}
                  {p.chapters > 0 && <Badge>{p.chapters} chapters</Badge>}
                  {p.license ? <Badge tone="neutral">{p.license}</Badge> : <Badge tone="amber">licence not set</Badge>}
                </div>
                <div className="mt-2.5">
                  <StageSteps stages={p.stages} />
                </div>
                {p.error && <p className="mt-1 truncate text-xs text-rose">{p.error}</p>}
              </div>
            </Link>
          ))}
        </div>

        <div className="space-y-4">
          <Panel title="New project">
            <label
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragging(false);
                setFile(e.dataTransfer.files[0] ?? null);
              }}
              className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-4 py-7 text-center transition ${dragging ? "border-violet bg-violet-soft" : "border-line hover:border-muted"}`}
            >
              <svg viewBox="0 0 24 24" className="mb-2 h-7 w-7 text-violet" fill="none" stroke="currentColor" strokeWidth="1.8">
                <path d="M12 16V4m0 0-4 4m4-4 4 4M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" strokeLinecap="round" />
              </svg>
              <span className="text-sm">{file ? file.name : "Drop a video or audio file"}</span>
              <span className="mt-1 text-[11px] text-muted">MP4, MOV, WEBM, MKV, MP3, WAV, M4A</span>
              <input type="file" accept="video/*,audio/*" className="hidden" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
            </label>
            <div className="mt-3 space-y-2">
              <label className="block text-[11px] font-medium text-muted">
                Licence of the source (required before export)
                <select value={license} onChange={(e) => setLicense(e.target.value)} className="mt-1 w-full rounded-lg border border-line bg-panel-2 px-2 py-1.5 text-sm text-ink">
                  {LICENSES.map((l) => (
                    <option key={l}>{l}</option>
                  ))}
                </select>
              </label>
              <label className="block text-[11px] font-medium text-muted">
                Attribution (added to descriptions and posts)
                <input
                  value={attribution}
                  onChange={(e) => setAttribution(e.target.value)}
                  placeholder="e.g. Video: Jane Doe, CC BY 4.0"
                  className="mt-1 w-full rounded-lg border border-line bg-panel-2 px-2 py-1.5 text-sm text-ink placeholder:text-muted/60"
                />
              </label>
            </div>
            <Button variant="primary" className="mt-3 w-full" disabled={!file || busy} onClick={upload}>
              {busy ? "Uploading…" : "Upload and process"}
            </Button>
            {error && <p className="mt-2 text-xs text-rose">{error}</p>}
          </Panel>
          {config && config.samples.length > 0 && (
            <Panel title="Or try a bundled sample">
              <div className="space-y-1.5">
                {config.samples.slice(0, 8).map((sample) => (
                  <button
                    key={sample.path}
                    disabled={busy}
                    onClick={() => importSample(sample)}
                    className="flex w-full items-center justify-between gap-2 rounded-lg border border-line-soft bg-panel-2 px-3 py-2 text-left text-sm hover:border-violet/50"
                  >
                    <span className="truncate">{sample.name}</span>
                    <span className="whitespace-nowrap text-[11px] text-violet">import →</span>
                  </button>
                ))}
              </div>
              <p className="mt-2 text-[11px] text-muted">Synthetic talks (CC0) and public-domain NASA briefings from the test set.</p>
            </Panel>
          )}
        </div>
      </div>
    </div>
  );
}
