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
import { CfgGroup, CfgRow, Stepper, SliderNum, RangeDual, Toggle, SeedInput, BarrierPicker, GridPreview } from "./ConfigControls";

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
  const [pendingChange, setPendingChange] = useState(false);

  useEffect(() => {
    try {
      setSensors(simulator.list_sensors() as RegistryEntry[]);
      setActions(simulator.list_actions() as RegistryEntry[]);
      setPendingChange(false);
    } catch { /* ignore */ }
  }, [simulator]);

  const toggleSensor = (entry: RegistryEntry) => {
    const next = !entry.enabled;
    simulator.set_sensor_enabled(entry.id, next);
    setSensors(s => s.map(x => x.id === entry.id ? { ...x, enabled: next } : x));
    setPendingChange(true);
  };

  const toggleAction = (entry: RegistryEntry) => {
    const next = !entry.enabled;
    simulator.set_action_enabled(entry.id, next);
    setActions(a => a.map(x => x.id === entry.id ? { ...x, enabled: next } : x));
    setPendingChange(true);
  };

  const enabledSensors = sensors.filter(s => s.enabled).length;
  const enabledActions = actions.filter(a => a.enabled).length;

  return (
    <>
      <DrawerHead title="Registry" />
      <div className="drawer-body">
        {pendingChange && (
          <div className="reg-pending">
            ⏱ Changes apply at the next generation — genome size will update then.
          </div>
        )}

        <div className="section-h">Sensors · {enabledSensors}/{sensors.length}</div>
        <ul className="reg-list">
          {sensors.map((s) => (
            <li key={s.id} className={s.enabled ? "" : "reg-disabled"}>
              <span className="reg-idx">{String(s.index).padStart(2, "0")}</span>
              <span className="reg-name">{s.name}</span>
              <span className="reg-kind">SEN</span>
              <button
                className={`reg-toggle ${s.enabled ? "on" : "off"}`}
                onClick={() => toggleSensor(s)}
                title={s.enabled ? "Disable sensor" : "Enable sensor"}
              >
                {s.enabled ? "ON" : "OFF"}
              </button>
            </li>
          ))}
        </ul>

        <div className="section-h">Actions · {enabledActions}/{actions.length}</div>
        <ul className="reg-list">
          {actions.map((a) => (
            <li key={a.id} className={a.enabled ? "" : "reg-disabled"}>
              <span className="reg-idx">{String(a.index).padStart(2, "0")}</span>
              <span className="reg-name">{a.name}</span>
              <span className="reg-kind">ACT</span>
              <button
                className={`reg-toggle ${a.enabled ? "on" : "off"}`}
                onClick={() => toggleAction(a)}
                title={a.enabled ? "Disable action" : "Enable action"}
              >
                {a.enabled ? "ON" : "OFF"}
              </button>
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
  const [cfg, setCfg] = useState<any>(() => JSON.parse(configJson));
  const [dirty, setDirty] = useState(false);
  const [showJson, setShowJson] = useState(false);
  const [jsonText, setJsonText] = useState(() => JSON.stringify(JSON.parse(configJson), null, 2));
  const [jsonError, setJsonError] = useState<string | null>(null);

  useEffect(() => {
    setCfg(JSON.parse(configJson));
    setJsonText(JSON.stringify(JSON.parse(configJson), null, 2));
    setDirty(false);
  }, [configJson]);

  const set = (k: string, v: any) => {
    setCfg((prev: any) => ({ ...prev, [k]: v }));
    setDirty(true);
  };

  const reset = () => {
    const base = JSON.parse(configJson);
    setCfg(base);
    setJsonText(JSON.stringify(base, null, 2));
    setJsonError(null);
    setDirty(false);
  };

  const apply = () => {
    if (showJson) {
      try {
        const parsed = JSON.parse(jsonText);
        onApplyConfig(jsonText);
        setCfg(parsed);
        setJsonError(null);
        setDirty(false);
      } catch (e) {
        setJsonError(e instanceof Error ? e.message : String(e));
      }
    } else {
      onApplyConfig(JSON.stringify(cfg));
      setDirty(false);
    }
  };

  // When toggling to JSON, sync form→text. When toggling to form, try to sync text→form.
  const toggleJson = () => {
    if (!showJson) {
      // form → JSON: serialise current cfg
      setJsonText(JSON.stringify(cfg, null, 2));
      setJsonError(null);
    } else {
      // JSON → form: try to parse
      try {
        setCfg(JSON.parse(jsonText));
        setJsonError(null);
      } catch {
        // leave cfg as-is; form will show last valid state
      }
    }
    setShowJson(j => !j);
  };

  const cells = cfg.size_x * cfg.size_y;
  const density = ((cfg.population / cells) * 100).toFixed(1);
  const totalSteps = cfg.steps_per_generation * cfg.max_generations;

  return (
    <>
      <div className="drawer-head">
        <span className="drawer-title">World config</span>
        <button className="drawer-action" onClick={toggleJson}>
          {showJson ? "FORM" : "JSON"}
        </button>
      </div>
      <div className="drawer-body cfg">
        <div className="cfg-paint">
          <div>
            <div className="cfg-paint-k">Painted barriers</div>
            <div className="cfg-paint-help">Persist across generations.</div>
          </div>
          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            <span className="cfg-paint-v">{paintedCount}</span>
            <button className="drawer-action" onClick={onClearPaint} disabled={paintedCount === 0}>CLEAR</button>
          </div>
        </div>

        {showJson ? (
          <div style={{ flex: 1, display: "flex", flexDirection: "column", padding: "12px 16px", gap: 8, minHeight: 0 }}>
            <div className="section-h">SimConfig · JSON</div>
            <textarea
              className="config-textarea"
              style={{ flex: 1, resize: "none", height: "auto" }}
              spellCheck={false}
              value={jsonText}
              onChange={(e) => { setJsonText(e.target.value); setDirty(true); }}
            />
            {jsonError && <p style={{ color: "var(--bad)", fontSize: 11, margin: 0 }}>{jsonError}</p>}
          </div>
        ) : (
          <>
            <CfgGroup title="World" summary={`${cfg.size_x}\u00b2 \u00b7 ${cells.toLocaleString()} cells`}>
              <CfgRow label="Grid size" help={`${cells.toLocaleString()} cells`}>
                <div className="cfg-grid-row">
                  <GridPreview size={cfg.size_x} pop={cfg.population} />
                  <div style={{flex: 1}}>
                    <Stepper value={cfg.size_x} min={32} max={512} step={32}
                             onChange={(v)=>{ set("size_x", v); set("size_y", v); }} suffix="px"/>
                    <div className="cfg-hint">Square world \u00b7 {cfg.size_x} \u00d7 {cfg.size_x}</div>
                  </div>
                </div>
              </CfgRow>
              <CfgRow label="Barriers">
                <BarrierPicker value={cfg.barrier_type} onChange={(v)=>set("barrier_type", v)}/>
              </CfgRow>
            </CfgGroup>
            <CfgGroup title="Population" summary={`${cfg.population.toLocaleString()} agents \u00b7 ${density}%`}>
              <CfgRow label="Population" help={`${density}% of grid`}>
                <SliderNum value={cfg.population} min={100} max={5000} step={50}
                           onChange={(v)=>set("population", v)}
                           markers={[500, 1000, 2500, 5000]}/>
              </CfgRow>
              <CfgRow label="Deterministic" help="Reproducible runs from a seed">
                <Toggle checked={cfg.deterministic} onChange={(v)=>set("deterministic", v)}/>
              </CfgRow>
              {cfg.deterministic && (
                <CfgRow label="RNG seed">
                  <SeedInput value={cfg.rng_seed} onChange={(v)=>set("rng_seed", v)}/>
                </CfgRow>
              )}
            </CfgGroup>
            <CfgGroup title="Time" summary={`${cfg.steps_per_generation} \u00d7 ${cfg.max_generations} = ${totalSteps.toLocaleString()} steps`}>
              <CfgRow label="Steps / generation">
                <SliderNum value={cfg.steps_per_generation} min={50} max={500} step={10}
                           onChange={(v)=>set("steps_per_generation", v)}
                           markers={[100, 200, 300, 500]}/>
              </CfgRow>
              <CfgRow label="Max generations">
                <SliderNum value={cfg.max_generations} min={10} max={1000} step={10}
                           onChange={(v)=>set("max_generations", v)}
                           markers={[100, 200, 500, 1000]}/>
              </CfgRow>
            </CfgGroup>
            <CfgGroup title="Genetics" summary={`${cfg.genome_initial_length_min === cfg.genome_initial_length_max ? cfg.genome_initial_length_min : `${cfg.genome_initial_length_min}\u2013${cfg.genome_initial_length_max}`} genes \u00b7 ${cfg.max_number_neurons} neurons`}>
              <CfgRow label="Genome length" help="Initial gene count per agent">
                <RangeDual
                  min={1} max={64}
                  low={cfg.genome_initial_length_min} high={cfg.genome_initial_length_max}
                  onChange={(lo, hi)=>{ set("genome_initial_length_min", lo); set("genome_initial_length_max", hi); }}
                />
              </CfgRow>
              <CfgRow label="Neurons" help="Hidden layer width">
                <Stepper value={cfg.max_number_neurons} min={1} max={20} step={1}
                         onChange={(v)=>set("max_number_neurons", v)}/>
              </CfgRow>
              <CfgRow label="Point mutation" help={`${(cfg.point_mutation_rate * 100).toFixed(2)}% per gene per offspring`}>
                <SliderNum value={cfg.point_mutation_rate} min={0} max={0.05} step={0.0005}
                           onChange={(v)=>set("point_mutation_rate", v)}
                           format={(v)=>`${(v*100).toFixed(2)}%`}
                           markers={[0, 0.005, 0.02, 0.05]}
                           markerLabels={["0%","0.5%","2%","5%"]}/>
              </CfgRow>
            </CfgGroup>
            <CfgGroup title="Behavior" summary={`resp ${cfg.responsiveness.toFixed(2)} \u00b7 probe ${cfg.long_probe_distance}`}>
              <CfgRow label="Responsiveness" help="0 = ignore neural output \u00b7 1 = fully driven">
                <SliderNum value={cfg.responsiveness} min={0} max={1} step={0.05}
                           onChange={(v)=>set("responsiveness", v)}
                           format={(v)=>v.toFixed(2)}/>
              </CfgRow>
              <CfgRow label="Population radius" help="Pop-density sensor sample radius">
                <SliderNum value={cfg.population_sensor_radius} min={1} max={10} step={0.5}
                           onChange={(v)=>set("population_sensor_radius", v)}
                           format={(v)=>`${v.toFixed(1)} cells`}/>
              </CfgRow>
              <CfgRow label="Signal radius" help="Pheromone sample radius">
                <SliderNum value={cfg.signal_sensor_radius} min={1} max={10} step={0.5}
                           onChange={(v)=>set("signal_sensor_radius", v)}
                           format={(v)=>`${v.toFixed(1)} cells`}/>
              </CfgRow>
              <CfgRow label="Long-probe distance" help="Forward line-of-sight in cells">
                <Stepper value={cfg.long_probe_distance} min={1} max={64} step={1}
                         onChange={(v)=>set("long_probe_distance", v)}/>
              </CfgRow>
              <CfgRow label="Kill enable" help="Agents can kill neighbours">
                <Toggle checked={cfg.kill_enable} onChange={(v)=>set("kill_enable", v)}/>
              </CfgRow>
            </CfgGroup>
          </>
        )}

        <div className="cfg-foot">
          <div className="cfg-foot-status">
            {dirty
              ? <><span className="cfg-dot dirty"/>Unsaved changes</>
              : <><span className="cfg-dot"/>In sync</>}
          </div>
          <div style={{display: "flex", gap: 8}}>
            <button className="btn-ghost" onClick={reset} disabled={!dirty}>Reset</button>
            <button className="btn-primary" onClick={apply}>Apply &amp; restart</button>
          </div>
        </div>
      </div>
    </>
  );
}
