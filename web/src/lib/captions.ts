/** The caption styles of recut/render/captions.py, mirrored for the live preview in the clip editor. */
import type { CaptionStyle, CropKey, Word } from "./types";

export interface StyleSpec {
  font: string;
  size: number; // in 1920-tall output pixels
  color: string;
  highlight: string;
  outline: string;
  box: string | null;
  maxWords: number;
  maxChars: number;
  uppercase: boolean;
  stripPunct: boolean;
  pop: boolean;
  karaoke?: boolean;
  oneWord?: boolean;
  lines?: number;
}

export const STYLES: Record<CaptionStyle, StyleSpec> = {
  bold: { font: "Poppins ExtraBold", size: 84, color: "#ffffff", highlight: "#ffe100", outline: "#000000", box: null, maxWords: 3, maxChars: 18, uppercase: true, stripPunct: true, pop: true },
  boxed: { font: "Archivo Black", size: 68, color: "#ffffff", highlight: "#a6f266", outline: "transparent", box: "rgba(18,24,32,0.69)", maxWords: 4, maxChars: 22, uppercase: false, stripPunct: false, pop: true },
  karaoke: { font: "Poppins ExtraBold", size: 74, color: "#ffffff", highlight: "#4de1ff", outline: "#141414", box: null, maxWords: 6, maxChars: 30, uppercase: false, stripPunct: false, pop: false, karaoke: true, lines: 2 },
  pop: { font: "Anton", size: 132, color: "#ffffff", highlight: "#ffe100", outline: "#000000", box: null, maxWords: 1, maxChars: 14, uppercase: true, stripPunct: true, pop: true, oneWord: true },
  minimal: { font: "Poppins ExtraBold", size: 54, color: "#ffffff", highlight: "#ffffff", outline: "rgba(0,0,0,0.6)", box: null, maxWords: 7, maxChars: 34, uppercase: false, stripPunct: false, pop: false, lines: 2 },
};

export interface TimedWord {
  index: number;
  start: number;
  end: number;
  text: string;
}

export function clipWords(words: Word[], start: number, end: number, overrides: Record<string, string>): TimedWord[] {
  const out: TimedWord[] = [];
  words.forEach((w, index) => {
    if (w.end <= start || w.start >= end) return;
    const text = (overrides[String(index)] ?? w.text).trim();
    if (!text) return;
    out.push({ index, start: Math.max(0, w.start - start), end: Math.min(end, w.end) - start, text });
  });
  return out;
}

export function groupWords(words: TimedWord[], spec: StyleSpec): TimedWord[][] {
  if (spec.oneWord) return words.map((w) => [w]);
  const lines = spec.lines ?? 1;
  const maxWords = spec.maxWords * (lines > 1 ? lines : 1);
  const maxChars = spec.maxChars * (lines > 1 ? lines : 1);
  const groups: TimedWord[][] = [];
  let current: TimedWord[] = [];
  let chars = 0;
  for (const w of words) {
    if (current.length) {
      const prev = current[current.length - 1];
      const pause = w.start - prev.end;
      if (
        current.length >= maxWords ||
        chars + 1 + w.text.length > maxChars ||
        /[.!?…]$/.test(prev.text) ||
        pause > 0.6 ||
        (/[,;:—–-]$/.test(prev.text) && current.length >= 2)
      ) {
        groups.push(current);
        current = [];
        chars = 0;
      }
    }
    current.push(w);
    chars += w.text.length + (chars ? 1 : 0);
  }
  if (current.length) groups.push(current);
  return groups;
}

export function display(text: string, spec: StyleSpec): string {
  let t = text;
  if (spec.stripPunct) t = t.replace(/^["'“‘(]+|["'”’),.;:!?…]+$/g, "") || text;
  return spec.uppercase ? t.toUpperCase() : t;
}

/** The group on screen at clip time t and the index (within it) of the word being spoken. */
export function activeAt(groups: TimedWord[][], t: number): { group: TimedWord[]; word: number } | null {
  for (let g = 0; g < groups.length; g++) {
    const group = groups[g];
    const next = groups[g + 1];
    let end = group[group.length - 1].end + 0.25;
    if (next && next[0].start - group[group.length - 1].end < 0.5) end = next[0].start;
    if (next) end = Math.min(end, next[0].start);
    if (t >= group[0].start && t < end) {
      let word = 0;
      for (let i = 0; i < group.length; i++) if (t >= group[i].start) word = i;
      return { group, word };
    }
  }
  return null;
}

export function interpolate(keys: CropKey[], t: number): number {
  if (!keys.length) return 0.5;
  if (t <= keys[0].t) return keys[0].cx;
  for (let i = 0; i < keys.length - 1; i++) {
    const a = keys[i];
    const b = keys[i + 1];
    if (a.t <= t && t <= b.t) return a.cx + ((b.cx - a.cx) * (t - a.t)) / Math.max(1e-6, b.t - a.t);
  }
  return keys[keys.length - 1].cx;
}
