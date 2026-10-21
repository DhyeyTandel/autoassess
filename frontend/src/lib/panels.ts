import type { PanelCounts, PanelName } from "../api/types";

export interface Region {
  panel: PanelName | null; // null = neutral, non-panel part
  key: string;
  kind: "rect" | "path";
  x?: number;
  y?: number;
  w?: number;
  h?: number;
  rx?: number;
  d?: string;
  active: boolean;
}

const rect = (x: number, y: number, w: number, h: number, rx = 0) => ({
  kind: "rect" as const, x, y, w, h, rx,
});
const path = (d: string) => ({ kind: "path" as const, d });

export function panelRegions(panels: Partial<PanelCounts>): Region[] {
  const on = (p: PanelName) => (panels[p] ?? 0) > 0;
  const mk = (panel: PanelName | null, key: string, shape: object): Region =>
    ({ panel, key, active: panel !== null && on(panel), ...shape }) as Region;
  return [
    mk(null, "roof", rect(60, 152, 80, 80)),
    mk(null, "rear_window", path("M62 234 L138 234 L146 262 L54 262 Z")),
    mk(null, "boot", rect(44, 264, 112, 70, 14)),
    mk("bonnet", "bonnet", rect(44, 40, 112, 70, 14)),
    mk("front_bumper", "front_bumper", rect(40, 18, 120, 22, 10)),
    mk("headlamp", "headlamp_l", rect(46, 22, 22, 10)),
    mk("headlamp", "headlamp_r", rect(132, 22, 22, 10)),
    mk("windshield", "windshield", path("M50 112 L150 112 L140 150 L60 150 Z")),
    mk("door", "door_fl", rect(32, 150, 26, 62)),
    mk("door", "door_fr", rect(142, 150, 26, 62)),
    mk("door", "door_rl", rect(32, 214, 26, 58)),
    mk("door", "door_rr", rect(142, 214, 26, 58)),
    mk("rear_bumper", "rear_bumper", rect(40, 336, 120, 22, 10)),
  ];
}

export function panelLabel(p: string): string {
  const s = p.replace(/_/g, " ");
  return s.charAt(0).toUpperCase() + s.slice(1);
}
