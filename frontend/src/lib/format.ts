export function prettifyLabel(raw: string): string {
  const s = raw.replace(/[_\s]+/g, " ").trim().toLowerCase();
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : s;
}

export function formatConfidence(c: number): string {
  return c.toFixed(2);
}

export function formatInt(n: number): string {
  return Math.round(n).toLocaleString("en-US");
}

export function formatSeconds(ms: number): string {
  return (ms / 1000).toFixed(1);
}
