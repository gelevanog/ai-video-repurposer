export function ts(seconds: number, millis = false): string {
  const s = Math.max(0, seconds);
  const whole = Math.floor(s);
  const h = Math.floor(whole / 3600);
  const m = Math.floor((whole % 3600) / 60);
  const sec = whole % 60;
  const base = h ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}` : `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
  return millis ? `${base}.${Math.floor((s - whole) * 10)}` : base;
}

export function duration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return m >= 60 ? `${Math.floor(m / 60)} h ${m % 60} min` : `${m} min ${s ? `${s} s` : ""}`.trim();
}

export function pct(value: number | null | undefined, digits = 0): string {
  return value === null || value === undefined ? "–" : `${value.toFixed(digits)}%`;
}

export const SPEAKER_COLORS = ["#a78bfa", "#34d399", "#fbbf24", "#60a5fa", "#f472b6"];

export function speakerColor(speaker: string): string {
  const n = Number(speaker.replace(/\D/g, "")) || 1;
  return SPEAKER_COLORS[(n - 1) % SPEAKER_COLORS.length];
}

export function scoreColor(score: number): string {
  if (score >= 75) return "#34d399";
  if (score >= 55) return "#fbbf24";
  return "#f87171";
}

/** X counts URLs as 23 characters and emoji / wide characters as 2. */
export function countChars(text: string, platform: string): number {
  if (platform !== "x") return [...text].length;
  let total = 0;
  const parts = text.split(/(https?:\/\/\S+)/);
  for (const part of parts) {
    if (/^https?:\/\//.test(part)) {
      total += 23;
      continue;
    }
    for (const ch of part) total += (ch.codePointAt(0) ?? 0) > 0xffff ? 2 : 1;
  }
  return total;
}
