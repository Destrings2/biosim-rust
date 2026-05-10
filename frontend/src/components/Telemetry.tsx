// Floating telemetry strip just above the playback bar — three sparkline
// cells showing live trends across recent generations.

import { IcClose } from "./Icons";
import { Sparkline } from "./Sparkline";

export interface TelemetryHistory {
  /** Survival rate per epoch (0..1). One sample per generation. */
  survival: number[];
  diversity: number[];
  alive: number[];
  /** Generation index of the LAST sample in each array. */
  latestGen: number;
}

interface Props {
  history: TelemetryHistory;
  population: number;
  onClose?: () => void;
}

const WINDOW = 24;

function tail<T>(xs: T[], n: number): T[] { return xs.slice(Math.max(0, xs.length - n)); }
function delta(xs: number[]): number {
  if (xs.length < 2) return 0;
  return xs[xs.length - 1] - xs[xs.length - 2];
}

export function Telemetry({ history, population, onClose }: Props) {
  const survival = tail(history.survival, WINDOW);
  const diversity = tail(history.diversity, WINDOW);
  const aliveAbs = tail(history.alive, WINDOW);
  const aliveNorm = aliveAbs.map((v) => population > 0 ? v / population : 0);

  const last = (xs: number[]) => xs.length ? xs[xs.length - 1] : 0;
  const fmtPct = (v: number) => `${(v * 100).toFixed(1)}%`;
  const fmtSign = (d: number, suffix = "") =>
    `${d >= 0 ? "+" : ""}${d.toFixed(2)}${suffix}`;
  const fmtSignPct = (d: number) =>
    `${d >= 0 ? "+" : ""}${(d * 100).toFixed(1)}`;

  const startGen = Math.max(0, history.latestGen - survival.length + 1);

  return (
    <div className="telemetry">
      <div className="tele-head">
        <span className="tele-title">Telemetry · last {survival.length} generation{survival.length === 1 ? "" : "s"}</span>
        <span style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span className="tele-gens">
            {survival.length > 0 ? `${startGen} → ${history.latestGen}` : "—"}
          </span>
          {onClose && (
            <button
              onClick={onClose}
              title="Hide telemetry"
              style={{
                background: "transparent", border: "none", color: "var(--muted)",
                padding: 0, display: "flex", alignItems: "center",
                cursor: "pointer",
              }}
            >
              <IcClose size={12}/>
            </button>
          )}
        </span>
      </div>
      <div className="tele-grid">
        <TeleCell
          k="Survival rate"
          v={survival.length ? fmtPct(last(survival)) : "—"}
          delta={survival.length > 1 ? fmtSignPct(delta(survival)) : "—"}
          dn={delta(survival) < 0}
          data={survival}
          accent
        />
        <TeleCell
          k="Diversity"
          v={diversity.length ? last(diversity).toFixed(2) : "—"}
          delta={diversity.length > 1 ? fmtSign(delta(diversity)) : "—"}
          dn={delta(diversity) < 0}
          data={diversity}
        />
        <TeleCell
          k="Alive @ gen end"
          v={aliveAbs.length ? String(Math.round(last(aliveAbs))) : "—"}
          delta={aliveAbs.length > 1 ? fmtSign(delta(aliveAbs)) : "—"}
          dn={delta(aliveAbs) < 0}
          data={aliveNorm}
          accent
        />
      </div>
    </div>
  );
}

interface CellProps { k: string; v: string; delta: string; dn?: boolean; data: number[]; accent?: boolean }
function TeleCell({ k, v, delta, dn, data, accent }: CellProps) {
  return (
    <div className="tele-cell">
      <div className="tele-k">{k}</div>
      <div className="tele-v-row">
        <div className="tele-v">{v}</div>
        <div className={`tele-delta ${dn ? "dn" : ""}`}>{delta}</div>
      </div>
      <div className="tele-spark">
        <Sparkline data={data} accent={accent} />
      </div>
    </div>
  );
}
