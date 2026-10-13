"use client";

import type { ReactNode } from "react";

import { scoreColor } from "@/lib/format";
import type { Stage } from "@/lib/types";

type Tone = "neutral" | "violet" | "lime" | "amber" | "rose" | "sky" | "mint";

const TONES: Record<Tone, string> = {
  neutral: "bg-panel-3 text-ink-soft border-line",
  violet: "bg-violet-soft text-violet-200 border-violet/30",
  lime: "bg-lime-soft text-lime border-lime/25",
  amber: "bg-amber-soft text-amber border-amber/25",
  rose: "bg-rose-soft text-rose border-rose/25",
  sky: "bg-sky-soft text-sky border-sky/25",
  mint: "bg-mint-soft text-mint border-mint/25",
};

export function Badge({ tone = "neutral", children, title }: { tone?: Tone; children: ReactNode; title?: string }) {
  return (
    <span title={title} className={`inline-flex items-center gap-1 whitespace-nowrap rounded-md border px-1.5 py-0.5 text-[11px] font-medium ${TONES[tone]}`}>
      {children}
    </span>
  );
}

export function Button({
  children,
  onClick,
  variant = "secondary",
  disabled,
  title,
  type = "button",
  className = "",
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "primary" | "secondary" | "ghost" | "success" | "danger";
  disabled?: boolean;
  title?: string;
  type?: "button" | "submit";
  className?: string;
}) {
  const styles = {
    primary: "bg-violet text-white hover:bg-violet/90 shadow-md shadow-violet/20",
    secondary: "bg-panel-3 text-ink border border-line hover:border-muted/60",
    ghost: "text-ink-soft hover:bg-panel-3",
    success: "bg-mint text-[#062018] hover:bg-mint/90 font-semibold",
    danger: "bg-rose-soft text-rose border border-rose/30 hover:bg-rose/20",
  }[variant];
  return (
    <button
      type={type}
      title={title}
      disabled={disabled}
      onClick={onClick}
      className={`inline-flex items-center justify-center gap-1.5 rounded-lg px-3 py-1.5 text-sm transition disabled:cursor-not-allowed disabled:opacity-40 ${styles} ${className}`}
    >
      {children}
    </button>
  );
}

export function Panel({ title, actions, children, className = "" }: { title?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`rounded-2xl border border-line-soft bg-panel ${className}`}>
      {(title || actions) && (
        <header className="flex items-center justify-between gap-3 border-b border-line-soft px-4 py-2.5">
          <h2 className="text-[13px] font-semibold tracking-wide text-ink-soft">{title}</h2>
          <div className="flex items-center gap-2">{actions}</div>
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  );
}

export function ScoreRing({ score, size = 44 }: { score: number; size?: number }) {
  const r = size / 2 - 4;
  const c = 2 * Math.PI * r;
  const color = scoreColor(score);
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="shrink-0">
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="#262b3b" strokeWidth="4" />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke={color}
        strokeWidth="4"
        strokeLinecap="round"
        strokeDasharray={`${(c * Math.min(100, score)) / 100} ${c}`}
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
      />
      <text x="50%" y="54%" textAnchor="middle" dominantBaseline="middle" fontSize={size * 0.32} fontWeight="700" fill="#eceef6">
        {Math.round(score)}
      </text>
    </svg>
  );
}

export function Meter({ label, value, max = 10 }: { label: string; value: number; max?: number }) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100));
  return (
    <div className="flex items-center gap-2 text-[11px]">
      <span className="w-[68px] shrink-0 text-muted">{label}</span>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-panel-3">
        <div className="h-full rounded-full" style={{ width: `${pct}%`, background: scoreColor(pct) }} />
      </div>
      <span className="mono w-7 text-right text-ink-soft">{max === 10 ? value.toFixed(0) : Math.round(value)}</span>
    </div>
  );
}

const STAGE_LABELS: Record<string, string> = {
  probe: "Probe",
  audio: "Audio",
  transcribe: "Transcribe",
  diarize: "Speakers",
  clips: "Find moments",
  chapters: "Chapters",
  posts: "Posts",
};

export function StageSteps({ stages }: { stages: Stage[] }) {
  return (
    <ol className="flex flex-wrap items-center gap-1.5">
      {stages.map((stage, i) => {
        const tone =
          stage.state === "done" || stage.state === "skipped"
            ? "border-mint/30 bg-mint-soft text-mint"
            : stage.state === "running"
              ? "border-violet/40 bg-violet-soft text-violet-200 animate-pulse"
              : stage.state === "failed"
                ? "border-rose/40 bg-rose-soft text-rose"
                : "border-line bg-panel-2 text-muted";
        return (
          <li key={stage.name} className="flex items-center gap-1.5" title={stage.message || stage.state}>
            <span className={`rounded-full border px-2 py-0.5 text-[11px] ${tone}`}>
              {stage.state === "done" ? "✓ " : ""}
              {STAGE_LABELS[stage.name] ?? stage.name}
              {stage.seconds >= 1 && stage.state === "done" ? <span className="ml-1 opacity-70">{stage.seconds.toFixed(0)}s</span> : null}
            </span>
            {i < stages.length - 1 && <span className="h-px w-3 bg-line" />}
          </li>
        );
      })}
    </ol>
  );
}

/** A tiny Markdown renderer for the generated show notes and blog posts (headings, lists, quotes, emphasis). */
export function Markdown({ text }: { text: string }) {
  const blocks = text.split(/\n{2,}/);
  const inline = (s: string) =>
    s
      .split(/(\*\*[^*]+\*\*|\*[^*]+\*)/)
      .map((part, i) =>
        part.startsWith("**") ? <strong key={i}>{part.slice(2, -2)}</strong> : part.startsWith("*") && part.length > 2 ? <em key={i}>{part.slice(1, -1)}</em> : part,
      );
  return (
    <div className="md text-sm">
      {blocks.map((block, i) => {
        const b = block.trim();
        if (!b) return null;
        if (b.startsWith("# ")) return <h1 key={i}>{inline(b.slice(2))}</h1>;
        if (b.startsWith("## ")) return <h2 key={i}>{inline(b.slice(3))}</h2>;
        if (b === "---") return <hr key={i} />;
        if (b.startsWith(">")) return <blockquote key={i}>{inline(b.replace(/^>\s?/gm, ""))}</blockquote>;
        if (/^[-*] /.test(b))
          return (
            <ul key={i}>
              {b.split("\n").map((line, j) => (
                <li key={j}>{inline(line.replace(/^[-*] /, ""))}</li>
              ))}
            </ul>
          );
        if (/^## /m.test(b)) {
          const [head, ...rest] = b.split("\n");
          return (
            <div key={i}>
              <h2>{inline(head.replace(/^## /, ""))}</h2>
              <p>{inline(rest.join(" "))}</p>
            </div>
          );
        }
        return <p key={i}>{inline(b.replace(/\n/g, " "))}</p>;
      })}
    </div>
  );
}

export function CharCounter({ chars, limit, target }: { chars: number; limit: number; target?: number }) {
  const over = chars > limit;
  const warn = !over && target !== undefined && chars > target;
  const pct = Math.min(100, (chars / limit) * 100);
  return (
    <div className="flex items-center gap-2 text-[11px]">
      <div className="h-1 w-20 overflow-hidden rounded-full bg-panel-3">
        <div className={`h-full ${over ? "bg-rose" : warn ? "bg-amber" : "bg-mint"}`} style={{ width: `${pct}%` }} />
      </div>
      <span className={`mono ${over ? "text-rose" : warn ? "text-amber" : "text-muted"}`}>
        {chars.toLocaleString()} / {limit.toLocaleString()}
      </span>
    </div>
  );
}
