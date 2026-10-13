"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { useApi } from "@/lib/api";
import type { AppConfig } from "@/lib/types";

const NAV = [
  { href: "/", label: "Projects", icon: "M4 6h16M4 12h16M4 18h10" },
  { href: "/evaluation", label: "Evaluation", icon: "M5 19V9m7 10V5m7 14v-7" },
];

export function Sidebar() {
  const path = usePathname();
  const { data: config } = useApi<AppConfig>("/api/config");
  return (
    <aside className="sticky top-0 flex h-screen w-[208px] shrink-0 flex-col border-r border-line-soft bg-panel px-3 py-5">
      <Link href="/" className="mb-7 flex items-center gap-2.5 px-2">
        <span className="grid h-8 w-8 place-items-center rounded-lg bg-gradient-to-br from-violet to-fuchsia-500 shadow-lg shadow-violet/30">
          <svg viewBox="0 0 24 24" className="h-4.5 w-4.5" fill="none" stroke="white" strokeWidth="2.4" strokeLinecap="round">
            <path d="M6 4v16M18 4v16M6 9h4M14 15h4" />
          </svg>
        </span>
        <span className="display text-lg tracking-tight">Recut</span>
      </Link>
      <nav className="space-y-1">
        {NAV.map((item) => {
          const active = item.href === "/" ? path === "/" || path.startsWith("/p/") : path.startsWith(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm ${active ? "bg-panel-3 text-ink" : "text-muted hover:bg-panel-2 hover:text-ink-soft"}`}
            >
              <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <path d={item.icon} />
              </svg>
              {item.label}
            </Link>
          );
        })}
      </nav>
      <div className="mt-auto space-y-2 rounded-xl border border-line-soft bg-panel-2 p-3 text-[11px] leading-relaxed text-muted">
        <div className="flex items-center gap-1.5 text-ink-soft">
          <span className="h-1.5 w-1.5 rounded-full bg-mint" /> Speech: whisper {config?.whisper_model ?? "small"} on CPU
        </div>
        <div className="break-all">
          Model: <span className="mono text-ink-soft">{config ? (config.llm_provider === "fake" ? "offline rules" : config.llm_model) : "…"}</span>
        </div>
        {config?.free_only && <div className="text-mint">free-only guard on</div>}
        <div>Nothing is published without your approval.</div>
      </div>
    </aside>
  );
}
