"use client";

import { useState } from "react";

import { Badge, Button, CharCounter, Markdown, Panel } from "@/components/ui";
import { API_URL, patch } from "@/lib/api";
import { countChars } from "@/lib/format";
import type { AppConfig, GroundingIssue, Outputs, Project, SocialPost } from "@/lib/types";

const PLATFORM_ICONS: Record<string, string> = { x: "𝕏", linkedin: "in", instagram: "◎", tiktok: "♪", threads: "@", facebook: "f" };

function Issues({ issues }: { issues: GroundingIssue[] }) {
  if (!issues.length) return null;
  return (
    <ul className="mt-2 space-y-1">
      {issues.map((issue, i) => (
        <li key={i} className="flex items-start gap-1.5 text-[11px] text-amber">
          <span>⚠</span>
          <span>{issue.message}</span>
        </li>
      ))}
    </ul>
  );
}

export function OutputsTab({ project, config, onChange }: { project: Project; config: AppConfig | null; onChange: (p: Partial<Project>) => void }) {
  const o = project.outputs;
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<"notes" | "blog">("notes");

  async function save(body: Record<string, unknown>) {
    setError(null);
    try {
      const result = await patch<{ outputs: Outputs; chapters_text: string }>(`/api/projects/${project.id}/outputs`, body);
      onChange({ outputs: result.outputs, chapters_text: result.chapters_text });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  const approvedClips = project.clips.filter((c) => c.status === "approved").length;
  const titleTarget = config?.youtube.title_target ?? 70;

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between rounded-2xl border border-line-soft bg-panel px-4 py-3">
        <div className="text-sm text-ink-soft">
          Drafted by <span className="mono text-ink">{o.generated_by || "–"}</span>. Every number and quote is checked against the transcript; flagged lines need a look before
          publishing.
        </div>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-1.5 whitespace-nowrap text-sm">
            <input type="checkbox" checked={o.approved} onChange={(e) => save({ approved: e.target.checked })} className="accent-mint" /> Texts approved
          </label>
          <a
            href={`${API_URL}/api/projects/${project.id}/export`}
            className={`whitespace-nowrap rounded-lg px-3 py-1.5 text-sm font-semibold ${approvedClips || o.approved ? "bg-mint text-[#062018]" : "pointer-events-none bg-panel-3 text-muted"}`}
          >
            ↓ Export bundle ({approvedClips} clips)
          </a>
        </div>
      </div>
      {error && <p className="rounded-lg border border-rose/30 bg-rose-soft px-3 py-2 text-xs text-rose">{error}</p>}

      <div className="grid grid-cols-[1fr_1fr] gap-5">
        <div className="space-y-5">
          <Panel title="YouTube" actions={<Badge tone={o.issues.length ? "amber" : "mint"}>{o.issues.length ? `${o.issues.length} flags` : "grounded"}</Badge>}>
            <label className="block text-[11px] text-muted">
              <span className="flex items-center justify-between">
                Title <CharCounter chars={o.seo_title.length} limit={config?.youtube.title ?? 100} target={titleTarget} />
              </span>
              <input
                key={o.seo_title}
                defaultValue={o.seo_title}
                onBlur={(e) => e.target.value !== o.seo_title && save({ seo_title: e.target.value })}
                className="mt-1 w-full rounded-lg border border-line bg-panel-2 px-2.5 py-2 text-[15px] font-semibold text-ink"
              />
            </label>
            <label className="mt-3 block text-[11px] text-muted">
              <span className="flex items-center justify-between">
                Description (chapters and credit included) <CharCounter chars={o.seo_description.length} limit={config?.youtube.description ?? 5000} />
              </span>
              <textarea
                key={o.seo_description}
                defaultValue={o.seo_description}
                rows={9}
                onBlur={(e) => e.target.value !== o.seo_description && save({ seo_description: e.target.value })}
                className="scroll-thin mt-1 w-full rounded-lg border border-line bg-panel-2 px-2.5 py-2 text-[13px] leading-relaxed text-ink-soft"
              />
            </label>
            <div className="mt-2 flex flex-wrap gap-1">
              {o.tags.map((t) => (
                <Badge key={t}>{t}</Badge>
              ))}
            </div>
            <Issues issues={o.issues} />
          </Panel>

          <Panel title="Chapters" actions={<Badge tone="violet">{project.chapters.length} chapters</Badge>}>
            <textarea
              key={project.chapters_text}
              defaultValue={project.chapters_text}
              rows={Math.max(4, project.chapters.length + 1)}
              onBlur={(e) => e.target.value !== project.chapters_text && save({ chapters_text: e.target.value })}
              className="mono w-full rounded-lg border border-line bg-panel-2 px-2.5 py-2 text-[12.5px] leading-6 text-ink"
            />
            <p className="mt-1 text-[11px] text-muted">YouTube format: first chapter at 00:00, at least three, each 10 s or longer (Recut keeps them 45 s or longer).</p>
          </Panel>
        </div>

        <Panel
          title={
            <span className="flex gap-1">
              {(["notes", "blog"] as const).map((v) => (
                <button key={v} onClick={() => setView(v)} className={`rounded-md px-2 py-0.5 ${view === v ? "bg-panel-3 text-ink" : "text-muted"}`}>
                  {v === "notes" ? "Show notes" : "Blog post"}
                </button>
              ))}
            </span>
          }
          actions={<span className="text-[11px] text-muted">{(view === "notes" ? o.show_notes : o.blog_post).split(/\s+/).filter(Boolean).length} words</span>}
        >
          <div className="scroll-thin max-h-[620px] overflow-y-auto pr-2">
            <Markdown text={view === "notes" ? o.show_notes : o.blog_post} />
          </div>
        </Panel>
      </div>

      <div>
        <h3 className="mb-2.5 text-[13px] font-semibold text-ink-soft">Social posts</h3>
        <div className="grid grid-cols-3 gap-4">
          {o.social.map((post) => (
            <SocialCard key={post.platform} post={post} config={config} onSave={(body) => save({ social: [{ platform: post.platform, ...body }] })} />
          ))}
        </div>
      </div>
    </div>
  );
}

function SocialCard({ post, config, onSave }: { post: SocialPost; config: AppConfig | null; onSave: (body: Partial<SocialPost>) => void }) {
  const [text, setText] = useState(post.text);
  const spec = config?.platforms[post.platform];
  const chars = countChars(text, post.platform);
  return (
    <div className={`flex flex-col rounded-2xl border bg-panel p-3.5 ${post.approved ? "border-mint/40" : "border-line-soft"}`}>
      <div className="mb-2 flex items-center justify-between">
        <span className="flex items-center gap-2 text-sm font-semibold">
          <span className="grid h-6 w-6 place-items-center rounded-md bg-panel-3 text-[12px]">{PLATFORM_ICONS[post.platform] ?? "•"}</span>
          {spec?.name ?? post.platform}
        </span>
        <CharCounter chars={chars} limit={post.limit} target={spec?.target} />
      </div>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        onBlur={() => text !== post.text && onSave({ text })}
        rows={6}
        className="scroll-thin flex-1 rounded-lg border border-line-soft bg-panel-2 px-2.5 py-2 text-[12.5px] leading-relaxed text-ink-soft"
      />
      <Issues issues={post.issues} />
      <div className="mt-2.5 flex items-center justify-between">
        {post.truncated ? <Badge tone="amber">cut to the limit by code</Badge> : <span />}
        <Button variant={post.approved ? "secondary" : "success"} className="!py-1 text-[12px]" onClick={() => onSave({ approved: !post.approved })} disabled={chars > post.limit}>
          {post.approved ? "Approved ✓" : "Approve"}
        </Button>
      </div>
    </div>
  );
}
