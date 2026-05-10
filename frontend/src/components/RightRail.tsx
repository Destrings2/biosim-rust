// Icon column + optional drawer on the right edge of the app.
// Drawers: Stats (live + sparklines), Challenge (current + params),
// Registry (sensors + actions), Config (JSON editor).

import { useEffect, useState } from "react";
import type { Simulator } from "../../pkg/biosim4_wasm.js";
import type { ChallengeConfig, ChallengeSchema, EpochResult, RegistryEntry, SimStats } from "../types";
import { IcChallenge, IcConfig, IcRegistry, IcStats } from "./Icons";
import { Sparkline } from "./Sparkline";
import { ChallengeArt, challengeArtKind } from "./ChallengeArt";
import type { TelemetryHistory } from "./Telemetry";

type DrawerId = "stats" | "challenge" | "registry" | "config" | null;

interface Props {
  simulator: Simulator;
  drawer: DrawerId;
  onDrawer: (d: DrawerId) => void;

  stats: SimStats | null;
  lastEpoch: EpochResult | null;
  history: TelemetryHistory;

  schemas: ChallengeSchema[];
  activeChallenge: ChallengeSchema | null;
  challengeParams: Record<string, unknown>;
  composition: "Any" | "All";
  onParamChange: (key: string, value: unknown) => void;
  onCompositionChange: (c: "Any" | "All") => void;
  onApplyChallenge: (cfg: ChallengeConfig) => void;
  onClearChallenge: () => void;
  onOpenPicker: () => void;

  paintedCount: number;
  onClearPaint: () => void;
  configJson: string;
  onApplyConfig: (json: string) => void;
}

export function RightRail(props: Props) {
  const items: { id: Exclude<DrawerId, null>; label: string; Ic: (p: { size?: number }) => JSX.Element }[] = [
    { id: "stats",     label: "Statistics", Ic: IcStats },
    { id: "challenge", label: "Challenge",  Ic: IcChallenge },
    { id: "registry",  label: "Registry",   Ic: IcRegistry },
    { id: "config",    label: "Config",     Ic: IcConfig },
  ];
  const { drawer, onDrawer } = props;

  return (
    <div className="rail">
      <div className="rail-icons">
        {items.map((it) => (
          <button
            key={it.id}
            className={`rail-btn ${drawer === it.id ? "active" : ""}`}
            onClick={() => onDrawer(drawer === it.id ? null : it.id)}
            title={it.label}
          >
            <it.Ic size={18}/>
          </button>
        ))}
        <div className="rail-spacer"/>
      </div>
      {drawer && (
        <div className="rail-drawer">
          {drawer === "stats"     && <StatsPanel {...props} />}
          {drawer === "challenge" && <ChallengePanel {...props} />}
          {drawer === "registry"  && <RegistryPanel simulator={props.simulator} />}
          {drawer === "config"    && <ConfigPanel {...props} />}
        </div>
      )}
    </div>
  );
}

// ── Drawer head ──────────────────────────────────────────────────────
function DrawerHead({ title, action, onAction, disabled }: { title: string; action?: string; onAction?: () => void; disabled?: boolean }) {
  return (
    <div className="drawer-head">
      <span className="drawer-title">{title}</span>
      {action && (
        <button className="drawer-action" onClick={onAction} disabled={disabled}>{action}</button>
      )}
    </div>
  );
}

// ── Stats ────────────────────────────────────────────────────────────
function StatsPanel({ stats, lastEpoch, history }: Pick<Props, "stats" | "lastEpoch" | "history">) {
  return (
    <>
      <DrawerHead title="Statistics" />
      <div className="drawer-body">
        <div className="stat-list">
          <div className="lbl">Generation</div>
          <div className="val">{stats ? String(stats.generation).padStart(4, "0") : "—"}</div>
          <div className="lbl">Step</div>
          <div className="val">{stats ? `${String(stats.sim_step).padStart(3, "0")} / ${stats.steps_per_generation}` : "—"}</div>
          <div className="lbl">Population</div>
          <div className="val">{stats ? stats.population.toLocaleString() : "—"}</div>
          <div className="lbl">Alive</div>
          <div className="val">{stats ? stats.alive_count.toLocaleString() : "—"}</div>
          <div className="lbl">Sensors</div>
          <div className="val">{stats ? stats.sensor_count : "—"}</div>
          <div className="lbl">Actions</div>
          <div className="val">{stats ? stats.action_count : "—"}</div>
        </div>

        {lastEpoch && (
          <>
            <div className="section-h">Last epoch · gen {lastEpoch.generation}</div>
            <div className="stat-list">
              <div className="lbl">Survivors</div>
              <div className="val">{lastEpoch.survivors.toLocaleString()}</div>
              <div className="lbl">Survival rate</div>
              <div className="val" style={{ color: "var(--accent)" }}>
                {(lastEpoch.survival_rate * 100).toFixed(1)}%
              </div>
              <div className="lbl">Diversity</div>
              <div className="val">{lastEpoch.diversity.toFixed(2)}</div>
            </div>
          </>
        )}

        <div className="section-h">Survival · {history.survival.length} gen</div>
        <Sparkline data={history.survival} accent />

        <div className="section-h">Diversity · {history.diversity.length} gen</div>
        <Sparkline data={history.diversity} />
      </div>
    </>
  );
}

// ── Challenge ────────────────────────────────────────────────────────
interface SchemaProperty {
  type: "number" | "boolean" | "string";
  minimum?: number;
  maximum?: number;
  default?: number | boolean | string;
  description?: string;
}

function schemaPropsOf(schema: Record<string, unknown>): Record<string, SchemaProperty> {
  return (schema.properties ?? {}) as Record<string, SchemaProperty>;
}

function ChallengePanel({
  activeChallenge, challengeParams, composition, onParamChange, onCompositionChange,
  onApplyChallenge, onClearChallenge, onOpenPicker,
}: Pick<Props, "activeChallenge" | "challengeParams" | "composition" | "onParamChange" | "onCompositionChange" | "onApplyChallenge" | "onClearChallenge" | "onOpenPicker">) {
  if (!activeChallenge) {
    return (
      <>
        <DrawerHead title="Challenge" action="GALLERY" onAction={onOpenPicker} />
        <div className="drawer-body">
          <p style={{ color: "var(--text-2)", fontSize: 12, lineHeight: 1.55 }}>
            No challenge is active. Selection passes everyone — nothing evolves. Open the gallery to pick a survival challenge.
          </p>
          <button className="btn-primary" style={{ marginTop: 14, width: "100%" }} onClick={onOpenPicker}>
            Open challenge gallery
          </button>
        </div>
      </>
    );
  }
  const props = schemaPropsOf(activeChallenge.schema);
  return (
    <>
      <DrawerHead title="Challenge" action="GALLERY" onAction={onOpenPicker} />
      <div className="drawer-body">
        <div className="param-block">
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
            <div style={{ width: 56, height: 36 }}>
              <ChallengeArt kind={challengeArtKind(activeChallenge.id)} />
            </div>
            <div>
              <div style={{ fontSize: 13, fontWeight: 600 }}>{activeChallenge.name}</div>
              <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
                {activeChallenge.id}
              </div>
            </div>
          </div>
          <p className="param-desc">{activeChallenge.description}</p>
          {Object.entries(props).map(([key, prop]) => (
            <ParamField key={key} name={key} prop={prop} value={challengeParams[key]}
                        onChange={(v) => onParamChange(key, v)} />
          ))}
        </div>

        <div className="section-h">Composition</div>
        <div className="seg" style={{ width: "100%" }}>
          <button className={composition === "Any" ? "on" : ""} onClick={() => onCompositionChange("Any")}>ANY</button>
          <button className={composition === "All" ? "on" : ""} onClick={() => onCompositionChange("All")}>ALL</button>
        </div>

        <div style={{ display: "flex", gap: 8, marginTop: 14 }}>
          <button
            className="btn-primary"
            style={{ flex: 1 }}
            onClick={() => onApplyChallenge({
              active: [activeChallenge.id],
              composition,
              params: { [activeChallenge.id]: challengeParams },
            })}
          >Apply</button>
          <button className="btn-ghost" onClick={onClearChallenge}>Clear</button>
        </div>
      </div>
    </>
  );
}

function ParamField({ name, prop, value, onChange }: {
  name: string; prop: SchemaProperty; value: unknown;
  onChange: (v: unknown) => void;
}) {
  if (prop.type === "boolean") {
    return (
      <label className="check-row" style={{ marginTop: 10 }}>
        <input
          type="checkbox"
          className="check"
          checked={Boolean(value)}
          onChange={(e) => onChange(e.target.checked)}
        />
        <span>{name}</span>
      </label>
    );
  }
  if (prop.type === "number") {
    const min = prop.minimum ?? 0;
    const max = prop.maximum ?? 1;
    const num = typeof value === "number" ? value : Number(prop.default ?? 0);
    return (
      <div className="field" style={{ marginBottom: 10 }}>
        <div className="field-lbl">
          <span>{name}</span>
          <span className="field-val">{num.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min={min} max={max} step={(max - min) / 100}
          value={num}
          onChange={(e) => onChange(Number(e.target.value))}
          style={{ width: "100%" }}
        />
      </div>
    );
  }
  // text fallback
  return (
    <div className="field" style={{ marginBottom: 10 }}>
      <div className="field-lbl"><span>{name}</span></div>
      <input
        type="text"
        value={String(value ?? "")}
        onChange={(e) => onChange(e.target.value)}
        style={{ width: "100%", background: "var(--bg-2)", border: "1px solid var(--line)", borderRadius: 4, padding: "6px 8px", color: "var(--text)" }}
      />
    </div>
  );
}

// ── Registry ─────────────────────────────────────────────────────────
function RegistryPanel({ simulator }: { simulator: Simulator }) {
  const [sensors, setSensors] = useState<RegistryEntry[]>([]);
  const [actions, setActions] = useState<RegistryEntry[]>([]);
  useEffect(() => {
    try {
      setSensors(simulator.list_sensors() as RegistryEntry[]);
      setActions(simulator.list_actions() as RegistryEntry[]);
    } catch { /* ignore */ }
  }, [simulator]);

  return (
    <>
      <DrawerHead title="Registry" />
      <div className="drawer-body">
        <div className="section-h">Sensors · {sensors.length}</div>
        <ul className="reg-list">
          {sensors.map((s) => (
            <li key={s.id}>
              <span className="reg-idx">{String(s.index).padStart(2, "0")}</span>
              <span className="reg-name">{s.name}</span>
              <span className="reg-kind">SEN</span>
            </li>
          ))}
        </ul>
        <div className="section-h">Actions · {actions.length}</div>
        <ul className="reg-list">
          {actions.map((a) => (
            <li key={a.id}>
              <span className="reg-idx">{String(a.index).padStart(2, "0")}</span>
              <span className="reg-name">{a.name}</span>
              <span className="reg-kind">ACT</span>
            </li>
          ))}
        </ul>
      </div>
    </>
  );
}

// ── Config ───────────────────────────────────────────────────────────
function ConfigPanel({
  paintedCount, onClearPaint, configJson, onApplyConfig,
}: Pick<Props, "paintedCount" | "onClearPaint" | "configJson" | "onApplyConfig">) {
  const [text, setText] = useState(configJson);
  const [error, setError] = useState<string | null>(null);

  return (
    <>
      <DrawerHead title="World config" />
      <div className="drawer-body">
        <div style={{ display: "flex", justifyContent: "space-between", padding: "10px 0", borderBottom: "1px solid var(--line)" }}>
          <span style={{ fontSize: 11, color: "var(--text-2)" }}>Painted barriers</span>
          <span style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text)" }}>{paintedCount}</span>
            <button className="drawer-action" onClick={onClearPaint} disabled={paintedCount === 0}>CLEAR</button>
          </span>
        </div>

        <div className="section-h">SimConfig · JSON</div>
        <textarea
          className="config-textarea"
          spellCheck={false}
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        {error && <p style={{ color: "var(--bad)", fontSize: 11, marginTop: 8 }}>{error}</p>}
        <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
          <button
            className="btn-primary"
            style={{ flex: 1 }}
            onClick={() => {
              try {
                JSON.parse(text);
                setError(null);
                onApplyConfig(text);
              } catch (e) {
                setError(e instanceof Error ? e.message : String(e));
              }
            }}
          >Apply &amp; reset</button>
          <button className="btn-ghost" onClick={() => { setText(configJson); setError(null); }}>
            Revert
          </button>
        </div>
      </div>
    </>
  );
}
