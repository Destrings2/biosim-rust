// Top-level shell: topbar, stage with framed canvas, right rail, modals.
// Ties together the wasm Simulator with all the redesigned components.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSimulator } from "./hooks/useSimulator";
import { SimCanvas } from "./components/SimCanvas";
import { TopBar } from "./components/TopBar";
import { FloatingToolbar, type Tool } from "./components/FloatingToolbar";
import { Playback } from "./components/Playback";
import { RightRail } from "./components/RightRail";
import { Telemetry, type TelemetryHistory } from "./components/Telemetry";
import { EmptyState } from "./components/EmptyState";
import { ChallengePickerModal } from "./components/ChallengePickerModal";
import { AgentInspector } from "./components/AgentInspector";
import type {
  ChallengeComposition, ChallengeConfig, ChallengeSchema, EpochResult, SimStats,
} from "./types";

const DEFAULT_CONFIG = {
  size_x: 128,
  size_y: 128,
  population: 1000,
  num_threads: 1,
  deterministic: true,
  rng_seed: 12345,
  signal_layers: 1,
  steps_per_generation: 200,
  max_generations: 200,
  genome_initial_length_min: 24,
  genome_initial_length_max: 24,
  genome_max_length: 300,
  max_number_neurons: 5,
  point_mutation_rate: 0.005,
  gene_insertion_deletion_rate: 0.0,
  deletion_ratio: 0.5,
  sexual_reproduction: false,
  choose_parents_by_fitness: true,
  kill_enable: false,
  responsiveness: 0.5,
  responsiveness_curve_k_factor: 2.0,
  population_sensor_radius: 2.5,
  signal_sensor_radius: 2.0,
  long_probe_distance: 16,
  short_probe_barrier_distance: 4,
  barrier_type: 0,
  genome_analysis_stride: 25,
  display_sample_genomes: 5,
  genome_comparison_method: 0,
  save_video: false,
  video_stride: 25,
};
const DEFAULT_CONFIG_JSON = JSON.stringify(DEFAULT_CONFIG, null, 2);

const HISTORY_CAP = 64;

export function App() {
  const { simulator, loading, error, reload, generation_token } =
    useSimulator(DEFAULT_CONFIG_JSON);

  // ── UI state
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(4);
  const [pixelSize, setPixelSize] = useState(4);
  const [tool, setTool] = useState<Tool>("inspect");
  const [drawer, setDrawer] = useState<"stats" | "challenge" | "registry" | "config" | null>("stats");
  const [showPicker, setShowPicker] = useState(false);
  const [inspectedAgentId, setInspectedAgentId] = useState<number | null>(null);
  const [showTelemetry, setShowTelemetry] = useState(true);

  // ── Sim state mirror
  const [stats, setStats] = useState<SimStats | null>(null);
  const [lastEpoch, setLastEpoch] = useState<EpochResult | null>(null);
  const [resetToken, setResetToken] = useState(0);
  const [paintedCount, setPaintedCount] = useState(0);
  const [history, setHistory] = useState<TelemetryHistory>({
    survival: [], diversity: [], alive: [], latestGen: 0,
  });

  // ── Challenge state
  const [schemas, setSchemas] = useState<ChallengeSchema[]>([]);
  const [activeChallengeId, setActiveChallengeId] = useState<string | null>(null);
  const [challengeParams, setChallengeParams] = useState<Record<string, unknown>>({});
  const [composition, setComposition] = useState<ChallengeComposition>("Any");

  const activeChallenge = useMemo(
    () => schemas.find((s) => s.id === activeChallengeId) ?? null,
    [schemas, activeChallengeId],
  );

  // FPS counter (smoothed, RAF-driven)
  const [fps, setFps] = useState(60);
  useEffect(() => {
    let raf = 0;
    let last = performance.now();
    let acc = 0;
    let frames = 0;
    const tick = (now: number) => {
      const dt = now - last;
      last = now;
      acc += dt;
      frames += 1;
      if (acc >= 500) {
        setFps((frames * 1000) / acc);
        acc = 0;
        frames = 0;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  // Pull schemas + initial stats whenever a fresh simulator instance shows up
  useEffect(() => {
    if (!simulator) return;
    try {
      setStats(simulator.get_stats() as SimStats);
      setPaintedCount(simulator.user_barrier_count());
      const list = simulator.get_challenge_schemas() as ChallengeSchema[];
      setSchemas(list);
      // Reset history on new simulator instance
      setHistory({ survival: [], diversity: [], alive: [], latestGen: 0 });
    } catch { /* mid-init */ }
  }, [simulator, generation_token, resetToken]);

  // ── Telemetry history: append on every epoch
  const onEpoch = useCallback((e: EpochResult) => {
    setLastEpoch(e);
    setHistory((h) => ({
      survival:  [...h.survival,  e.survival_rate].slice(-HISTORY_CAP),
      diversity: [...h.diversity, e.diversity].slice(-HISTORY_CAP),
      alive:     [...h.alive,     e.survivors].slice(-HISTORY_CAP),
      latestGen: e.generation,
    }));
  }, []);

  // ── Playback handlers
  const onTogglePlay = useCallback(() => setRunning((r) => !r), []);
  const onStep = useCallback(() => {
    if (!simulator || running) return;
    if (simulator.sim_step() >= simulator.steps_per_generation()) {
      const e = simulator.spawn_next_generation() as EpochResult;
      onEpoch(e);
    } else {
      simulator.step();
    }
    setStats(simulator.get_stats() as SimStats);
    setResetToken((t) => t + 1);
  }, [simulator, running, onEpoch]);
  const onStepGen = useCallback(() => {
    if (!simulator || running) return;
    // If we're already parked at the end of a generation, advance into the
    // next one (selection + reproduction). Otherwise run the remaining
    // steps in this generation so we end up parked at the boundary.
    if (simulator.sim_step() >= simulator.steps_per_generation()) {
      const e = simulator.spawn_next_generation() as EpochResult;
      onEpoch(e);
    } else {
      simulator.step_generation();
    }
    setStats(simulator.get_stats() as SimStats);
    setResetToken((t) => t + 1);
  }, [simulator, running, onEpoch]);
  const onRunEpoch = useCallback(() => {
    if (!simulator || running) return;
    const e = simulator.run_epoch() as EpochResult;
    onEpoch(e);
    setStats(simulator.get_stats() as SimStats);
    setResetToken((t) => t + 1);
  }, [simulator, running, onEpoch]);
  const onReset = useCallback(() => {
    if (!simulator) return;
    setRunning(false);
    setLastEpoch(null);
    setHistory({ survival: [], diversity: [], alive: [], latestGen: 0 });
    simulator.reset();
    setStats(simulator.get_stats() as SimStats);
    setResetToken((t) => t + 1);
  }, [simulator]);

  // ── Challenge handlers
  const applyChallenge = useCallback((cfg: ChallengeConfig) => {
    if (!simulator) return;
    try {
      simulator.set_challenge(JSON.stringify(cfg));
      setActiveChallengeId(cfg.active[0] ?? null);
      const pickedParams = cfg.active[0] ? (cfg.params[cfg.active[0]] ?? {}) : {};
      setChallengeParams(pickedParams);
      setComposition(typeof cfg.composition === "string" ? cfg.composition : "Any");
      // Reset epoch history so the trend graph shows the new selection cleanly.
      onReset();
    } catch (err) { console.error("set_challenge failed:", err); }
  }, [simulator, onReset]);

  const clearChallenge = useCallback(() => {
    if (!simulator) return;
    try {
      simulator.set_challenge(JSON.stringify({ active: [], composition: "Any", params: {} }));
      setActiveChallengeId(null);
      setChallengeParams({});
    } catch (err) { console.error("set_challenge failed:", err); }
  }, [simulator]);

  const onApplyConfig = useCallback((json: string) => {
    setRunning(false);
    setLastEpoch(null);
    setStats(null);
    void reload(json);
  }, [reload]);

  const clearPaintedBarriers = useCallback(() => {
    if (!simulator) return;
    simulator.clear_user_barriers();
    setPaintedCount(0);
    setResetToken((t) => t + 1);
  }, [simulator]);

  // ── Keyboard shortcuts
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA")) return;
      if (e.key === " ")             { e.preventDefault(); setRunning((r) => !r); }
      else if (e.key === "i" || e.key === "1") setTool("inspect");
      else if (e.key === "b" || e.key === "2") setTool("barrier");
      else if (e.key === "k" || e.key === "3") setTool("kill");
      else if (e.key === "r" || e.key === "4") setTool("reproduce");
      else if (e.key === "c")        setShowPicker(true);
      else if (e.key === "Escape")   { setShowPicker(false); setInspectedAgentId(null); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // ── Empty-state condition: gen 0 step 0 and not running yet
  const isEmpty = !running && stats?.generation === 0 && stats?.sim_step === 0;

  // ── Canvas frame size
  //
  // PX slider is a true multiplier: target = pixelSize × size_x.
  // Then we cap by viewport so the frame never overflows the stage. We also
  // enforce a sensible minimum so the floating toolbar (which is wider than
  // its frame at small sizes) doesn't overflow the canvas's left/right edges.
  const stageRef = useRef<HTMLDivElement>(null);
  const [stageBox, setStageBox] = useState<{ w: number; h: number }>({ w: 800, h: 800 });
  useEffect(() => {
    const update = () => {
      const el = stageRef.current;
      if (!el) return;
      setStageBox({ w: el.clientWidth, h: el.clientHeight });
    };
    update();
    const ro = new ResizeObserver(update);
    if (stageRef.current) ro.observe(stageRef.current);
    return () => ro.disconnect();
  }, [drawer]);

  const frameWidth = useMemo(() => {
    const sx = simulator?.size_x() ?? 128;
    // Headroom for the floating toolbar (54px above) and playback (52px below)
    // and a bit of breathing room.
    const vSlack = 200;
    const hSlack = 60;
    const target = pixelSize * sx;
    const maxByH = Math.max(360, stageBox.h - vSlack);
    const maxByW = Math.max(360, stageBox.w - hSlack);
    // Minimum 480px so the floating toolbar — which is ~470px wide — fits.
    return Math.max(480, Math.min(target, maxByH, maxByW));
  }, [pixelSize, simulator, stageBox]);

  // Expose canvas width via CSS var for the telemetry overlay alignment
  useEffect(() => {
    document.documentElement.style.setProperty("--canvas-w", `${frameWidth}px`);
  }, [frameWidth]);

  if (loading && !simulator) {
    return <div className="loading">Loading wasm module…</div>;
  }
  if (error) {
    return (
      <div className="error">
        <h2>Failed to load simulator</h2>
        <pre>{error}</pre>
      </div>
    );
  }
  if (!simulator) return null;

  return (
    <div className="shell">
      <TopBar
        stats={stats}
        fps={fps}
        speed={speed}
        activeChallenge={activeChallenge}
        showTelemetry={showTelemetry}
        onToggleTelemetry={() => setShowTelemetry((v) => !v)}
        onChallengeClick={() => setShowPicker(true)}
      />

      <div className="body">
        <div className="stage" ref={stageRef}>
          <div className="canvas-frame" style={{ width: frameWidth }}>
            {/* Frame chrome */}
            <Crosshair pos="tl"/><Crosshair pos="tr"/>
            <Crosshair pos="bl"/><Crosshair pos="br"/>
            <div className="frame-label tl">GRID {simulator.size_x()} × {simulator.size_y()}</div>
            <div className="frame-label tr">{paintedCount > 0 ? `${paintedCount} PAINTED` : "PROCEDURAL"}</div>
            <div className="frame-label bl">PIXEL · {pixelSize}×</div>
            <div className="frame-label br">{running ? "RUNNING" : "PAUSED"} · {speed}× SPF</div>

            <div className="canvas-bevel"/>
            <SimCanvas
              simulator={simulator}
              running={running}
              speed={speed}
              pixelSize={pixelSize}
              onStats={setStats}
              onEpoch={onEpoch}
              resetToken={resetToken}
              tool={tool}
              onAgentClick={setInspectedAgentId}
              onWorldChange={() => {
                try {
                  setStats(simulator.get_stats() as SimStats);
                  setPaintedCount(simulator.user_barrier_count());
                } catch { /* ignore */ }
              }}
            />

            <FloatingToolbar active={tool} onChange={setTool} />

            <Playback
              paused={!running}
              onTogglePlay={onTogglePlay}
              onStep={onStep}
              onStepGen={onStepGen}
              onRunEpoch={onRunEpoch}
              onReset={onReset}
              speed={speed}
              onSpeedChange={setSpeed}
              pixel={pixelSize}
              onPixelChange={setPixelSize}
            />

            {showTelemetry && stats && !isEmpty && (
              <Telemetry
                history={history}
                population={stats.population}
                onClose={() => setShowTelemetry(false)}
              />
            )}

            {isEmpty && (
              <EmptyState
                challenge={activeChallenge}
                population={stats?.population ?? 0}
                sensorCount={stats?.sensor_count ?? 0}
                actionCount={stats?.action_count ?? 0}
                stepsPerGen={stats?.steps_per_generation ?? 0}
                onPlay={() => setRunning(true)}
                onChooseChallenge={() => setShowPicker(true)}
              />
            )}
          </div>
        </div>

        <RightRail
          simulator={simulator}
          drawer={drawer}
          onDrawer={setDrawer}
          stats={stats}
          lastEpoch={lastEpoch}
          history={history}
          schemas={schemas}
          activeChallenge={activeChallenge}
          challengeParams={challengeParams}
          composition={composition === "Any" || composition === "All" ? composition : "Any"}
          onParamChange={(k, v) => setChallengeParams((p) => ({ ...p, [k]: v }))}
          onCompositionChange={setComposition}
          onApplyChallenge={applyChallenge}
          onClearChallenge={clearChallenge}
          onOpenPicker={() => setShowPicker(true)}
          paintedCount={paintedCount}
          onClearPaint={clearPaintedBarriers}
          configJson={DEFAULT_CONFIG_JSON}
          onApplyConfig={onApplyConfig}
        />
      </div>

      {showPicker && (
        <ChallengePickerModal
          schemas={schemas}
          activeId={activeChallengeId}
          onClose={() => setShowPicker(false)}
          onApply={applyChallenge}
        />
      )}
      {inspectedAgentId !== null && (
        <AgentInspector
          agentId={inspectedAgentId}
          simulator={simulator}
          onClose={() => setInspectedAgentId(null)}
        />
      )}
    </div>
  );
}

function Crosshair({ pos }: { pos: "tl" | "tr" | "bl" | "br" }) {
  return (
    <svg className={`cross ${pos}`} viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1">
      <path d="M0 7 H14 M7 0 V14"/>
    </svg>
  );
}
