"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { API_URL } from "@/lib/api";
import { STYLES, activeAt, clipWords, display, groupWords, interpolate } from "@/lib/captions";
import { ts } from "@/lib/format";
import type { Clip, CropPlan, SourceInfo, Word } from "@/lib/types";

const W = 300;
const H = Math.round((W * 16) / 9);
const OUT_H = 1920;

export interface PreviewHandle {
  seek: (sourceTime: number) => void;
}

/** A 9:16 phone frame: the source video framed the way the render will frame it (tracked crop, centre crop or full
 * frame over a blurred fill) with the caption style, word highlight and title card drawn live over it. */
export function PhonePreview({
  projectId,
  clip,
  words,
  source,
  crop,
  mode,
  onTime,
  handleRef,
}: {
  projectId: string;
  clip: Clip;
  words: Word[];
  source: SourceInfo;
  crop: CropPlan | null;
  mode: "live" | "rendered";
  onTime?: (sourceTime: number) => void;
  handleRef?: React.MutableRefObject<PreviewHandle | null>;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const [t, setT] = useState(clip.start);
  const [playing, setPlaying] = useState(false);
  const rendered = mode === "rendered" && clip.render;

  const timed = useMemo(() => clipWords(words, clip.start, clip.end, clip.caption_overrides), [words, clip.start, clip.end, clip.caption_overrides]);
  const spec = STYLES[clip.caption_style];
  const groups = useMemo(() => groupWords(timed, spec), [timed, spec]);

  const seek = useCallback(
    (sourceTime: number) => {
      const el = video.current;
      if (!el) return;
      const clamped = Math.min(Math.max(sourceTime, clip.start), clip.end - 0.05);
      el.currentTime = rendered ? clamped - clip.start : clamped;
      setT(clamped);
    },
    [clip.start, clip.end, rendered],
  );

  useEffect(() => {
    if (handleRef) handleRef.current = { seek };
  }, [handleRef, seek]);

  useEffect(() => {
    const el = video.current;
    if (!el) return;
    const start = () => {
      el.currentTime = rendered ? 0 : clip.start;
      setT(clip.start);
    };
    if (el.readyState >= 1) start();
    else el.addEventListener("loadedmetadata", start, { once: true });
    setPlaying(false);
    el.pause();
  }, [clip.id, clip.start, rendered]);

  useEffect(() => {
    let frame = 0;
    const tick = () => {
      const el = video.current;
      if (el) {
        const now = el.currentTime + (rendered ? clip.start : 0);
        if (now >= clip.end - 0.02 && !el.paused) {
          el.pause();
          setPlaying(false);
        }
        setT(now);
        onTime?.(now);
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [clip.end, clip.start, rendered, onTime]);

  function toggle() {
    const el = video.current;
    if (!el) return;
    if (el.paused) {
      const now = el.currentTime + (rendered ? clip.start : 0);
      if (now < clip.start || now >= clip.end - 0.1) seek(clip.start);
      void el.play();
      setPlaying(true);
    } else {
      el.pause();
      setPlaying(false);
    }
  }

  const layout = crop?.layout ?? (clip.layout === "auto" ? "center" : clip.layout === "track" ? "center" : clip.layout);
  const rel = t - clip.start;
  const srcRatio = source.width && source.height ? source.width / source.height : 16 / 9;
  let videoStyle: React.CSSProperties;
  if (layout === "fit" || !source.has_video) {
    const h = W / srcRatio;
    videoStyle = { width: W, height: h, left: 0, top: (H - h) / 2 - (120 / OUT_H) * H };
  } else {
    const vw = H * srcRatio;
    const cx = layout === "track" && crop ? interpolate(crop.keys, rel) : 0.5;
    const left = Math.min(0, Math.max(W - vw, W / 2 - cx * vw));
    videoStyle = { width: vw, height: H, left, top: 0 };
  }

  const active = activeAt(groups, rel);
  const scale = H / OUT_H;
  const marginV = (layout === "fit" ? 470 : 560) * scale;
  const titleTop = (layout === "fit" ? 300 : 250) * scale;
  const showTitle = clip.title_card && rel < 3 && (clip.title_card_text || clip.title);

  return (
    <div className="flex flex-col items-center">
      <div className="phone relative overflow-hidden bg-black" style={{ width: W, height: H }}>
        {rendered ? (
          <video ref={video} src={`${API_URL}/api/projects/${projectId}/clips/${clip.id}/video?v=${clip.render?.settings_hash}`} className="h-full w-full" playsInline preload="auto" />
        ) : (
          <>
            {(layout === "fit" || !source.has_video) && (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={`${API_URL}/api/projects/${projectId}/thumb`}
                alt=""
                className="absolute inset-0 h-full w-full scale-125 object-cover opacity-70 blur-xl"
                onError={(e) => (e.currentTarget.style.display = "none")}
              />
            )}
            <video ref={video} src={`${API_URL}/api/projects/${projectId}/media`} className="absolute max-w-none" style={videoStyle} playsInline preload="auto" />
            {showTitle && (
              <div className="absolute inset-x-0 flex justify-center px-6" style={{ top: titleTop }}>
                <span
                  className="rounded-[4px] bg-white px-2 py-1 text-center leading-tight text-[#151515]"
                  style={{ fontFamily: "Poppins ExtraBold", fontSize: 62 * scale, boxDecorationBreak: "clone", WebkitBoxDecorationBreak: "clone" }}
                >
                  {clip.title_card_text || clip.title}
                </span>
              </div>
            )}
            {active && (
              <div className="pointer-events-none absolute inset-x-0 flex justify-center px-5 text-center" style={{ bottom: marginV }}>
                <span
                  style={{
                    fontFamily: spec.font,
                    fontSize: spec.size * scale,
                    lineHeight: 1.15,
                    color: spec.color,
                    background: spec.box ?? undefined,
                    padding: spec.box ? `${4 * scale * 3}px ${8 * scale * 3}px` : undefined,
                    borderRadius: spec.box ? 6 : undefined,
                    WebkitTextStroke: spec.box ? undefined : `${(spec.size > 100 ? 8 : 6) * scale}px ${spec.outline}`,
                    paintOrder: "stroke fill",
                    textShadow: spec.box ? undefined : "0 2px 6px rgba(0,0,0,0.45)",
                    maxWidth: "100%",
                  }}
                >
                  {active.group.map((w, i) => {
                    const isActive = i === active.word;
                    const spoken = i <= active.word;
                    const color = spec.karaoke ? (spoken ? spec.highlight : spec.color) : spec.pop && isActive ? spec.highlight : spec.oneWord && /\d/.test(w.text) ? spec.highlight : spec.color;
                    return (
                      <span key={w.index}>
                        <span className="caption-word" style={{ color, transform: spec.pop && isActive && !spec.oneWord ? "scale(1.1)" : undefined }}>
                          {display(w.text, spec)}
                        </span>
                        {i < active.group.length - 1 ? " " : ""}
                      </span>
                    );
                  })}
                </span>
              </div>
            )}
          </>
        )}
        <button onClick={toggle} className="absolute inset-0 flex items-center justify-center" aria-label={playing ? "Pause" : "Play"}>
          {!playing && (
            <span className="grid h-14 w-14 place-items-center rounded-full bg-black/45 backdrop-blur">
              <svg viewBox="0 0 24 24" className="ml-1 h-6 w-6" fill="white">
                <path d="M7 4v16l13-8z" />
              </svg>
            </span>
          )}
        </button>
        <div className="absolute left-3 top-3 rounded-md bg-black/55 px-1.5 py-0.5 text-[10px] font-medium text-white/90">
          {rendered ? "Rendered MP4" : layout === "track" ? "Live preview · face tracking" : layout === "fit" ? "Live preview · full frame" : "Live preview · centre crop"}
        </div>
      </div>
      <div className="mt-4 flex w-[300px] items-center gap-2">
        <span className="mono text-[11px] text-muted">{ts(Math.max(0, t - clip.start), true)}</span>
        <input
          type="range"
          className="scrub flex-1"
          min={clip.start}
          max={clip.end}
          step={0.05}
          value={Math.min(Math.max(t, clip.start), clip.end)}
          onChange={(e) => seek(Number(e.target.value))}
        />
        <span className="mono text-[11px] text-muted">{ts(clip.end - clip.start, true)}</span>
      </div>
    </div>
  );
}
