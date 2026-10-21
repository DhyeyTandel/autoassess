import type { PanelCounts, PanelName } from "../api/types";
import { panelLabel, panelRegions } from "../lib/panels";

const ORDER: PanelName[] = ["windshield", "door", "bonnet", "front_bumper", "headlamp", "rear_bumper"];

export function PanelMap({ panels, unassigned }: { panels: PanelCounts; unassigned: number }) {
  const regions = panelRegions(panels);
  const affected = ORDER.filter((k) => panels[k] > 0).map((k) => `${panelLabel(k)} ${panels[k]}`);
  const aria =
    "Top view of a car. " +
    (affected.length ? `Affected panels: ${affected.join(", ")}.` : "No panels matched.");
  return (
    <section className="card" aria-labelledby="panelmap-h">
      <h2 className="mono-label" id="panelmap-h">PANEL MAP</h2>
      <div className="panelmap">
        <svg viewBox="0 0 200 380" role="img" aria-label={aria} className="carmap">
          {regions.map((r) => {
            const cls = r.active ? "region on" : "region";
            return r.kind === "rect" ? (
              <rect key={r.key} className={cls} x={r.x} y={r.y} width={r.w} height={r.h} rx={r.rx} />
            ) : (
              <path key={r.key} className={cls} d={r.d} />
            );
          })}
        </svg>
        <ul className="panel-list">
          {ORDER.map((k) => (
            <li key={k} className={panels[k] > 0 ? "hit" : ""}>
              <span>{panelLabel(k)}</span>
              <span className="mono">{panels[k]}</span>
            </li>
          ))}
          <li className={unassigned > 0 ? "hit" : ""}>
            <span>No panel matched</span>
            <span className="mono">{unassigned}</span>
          </li>
        </ul>
      </div>
    </section>
  );
}
