import type { AssessResult, Decision, Severity } from "../api/types";
import { formatConfidence, formatInt, prettifyLabel } from "../lib/format";
import { PanelMap } from "./PanelMap";

const DECISION_COLOR: Record<Decision, string> = {
  total_loss_review: "var(--severe)",
  human_review: "var(--moderate)",
  auto_approve: "var(--minor)",
};

export function Result({ r, filename }: { r: AssessResult; filename: string }) {
  const t = r.triage;
  const counts: [Severity, number][] = [
    ["severe", t.n_severe],
    ["moderate", t.n_moderate],
    ["minor", t.n_minor],
  ];
  return (
    <>
      <div className="result-grid">
        <section className="card frame" aria-label="Annotated photo">
          <div className="frame-head mono">
            <span className="trunc">FRAME {filename} · {r.image.width} × {r.image.height}</span>
            <span className="muted">MODELS · CarDD + VehiDE + panels</span>
          </div>
          <img className="overlay" src={r.overlay} alt={`Photo with ${t.n_instances} damage findings outlined`} />
          <dl className="readout">
            {counts.map(([sev, n]) => (
              <div key={sev} className="readout-cell">
                <dd className="bignum" style={{ color: `var(--${sev})` }}>{n}</dd>
                <dt className="small-label">{sev}</dt>
              </div>
            ))}
          </dl>
        </section>
        <div className="side">
          <section
            className="card routing"
            style={{ borderTop: `3px solid ${DECISION_COLOR[t.decision]}` }}
            aria-labelledby="routing-h"
          >
            <h2 className="mono-label" id="routing-h">ROUTING</h2>
            <p className="decision" style={{ color: DECISION_COLOR[t.decision] }}>{t.label}</p>
            <p className="secondary">
              {t.n_instances} findings · {formatInt(t.total_damaged_area_px)} px damaged. Hand-set
              heuristic, not an adjuster decision.
            </p>
          </section>
          <PanelMap panels={r.panels} unassigned={r.unassigned} />
        </div>
      </div>
      <section className="card log" aria-labelledby="log-h">
        <h2 className="mono-label log-head" id="log-h">FINDINGS LOG · {r.instances.length}</h2>
        <div className="table-scroll" tabIndex={0} role="region" aria-labelledby="log-h">
          <table>
            <thead>
              <tr>
                <th scope="col">#</th>
                <th scope="col">Damage</th>
                <th scope="col">Panel</th>
                <th scope="col">Grade</th>
                <th scope="col" className="num">Confidence</th>
              </tr>
            </thead>
            <tbody>
              {r.instances.map((i, idx) => (
                <tr key={idx}>
                  <td className="mono">{idx + 1}</td>
                  <td>{prettifyLabel(i.damage)}</td>
                  <td>{i.panel ? prettifyLabel(i.panel) : "—"}</td>
                  <td style={{ color: `var(--${i.severity})` }} className="grade">{prettifyLabel(i.severity)}</td>
                  <td className="mono num">{formatConfidence(i.confidence)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}
