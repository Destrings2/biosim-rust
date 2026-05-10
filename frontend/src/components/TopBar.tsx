// Top status bar: brand mark + live stats + telemetry toggle + active-challenge chip.

import type { ChallengeSchema, SimStats } from "../types";
import { IcKeyboard, IcTelemetry } from "./Icons";

interface Props {
  stats: SimStats | null;
  fps: number;
  speed: number;
  activeChallenge: ChallengeSchema | null;
  showTelemetry: boolean;
  onToggleTelemetry: () => void;
  onChallengeClick: () => void;
}

export function TopBar({
  stats, fps, speed, activeChallenge, showTelemetry, onToggleTelemetry, onChallengeClick,
}: Props) {
  const gen   = stats ? String(stats.generation).padStart(4, "0") : "—";
  const step  = stats ? `${String(stats.sim_step).padStart(3, "0")} / ${stats.steps_per_generation}` : "—";
  const alive = stats ? `${stats.alive_count} / ${stats.population}` : "—";

  return (
    <div className="topbar">
      <div className="brand">
        <div className="brand-mark"/>
        <span className="brand-name">BIOSIM</span>
        <span className="brand-tag">v4 · wasm</span>
      </div>
      <div className="topbar-stats">
        <Stat k="GEN"   v={gen} />
        <Stat k="STEP"  v={step} dim />
        <Stat k="ALIVE" v={alive} accent />
        <Stat k="SPF"   v={`${speed}×`} dim />
        <Stat k="FPS"   v={fps.toFixed(1)} dim />
      </div>
      <div className="topbar-right">
        <button
          className={`top-icon ${showTelemetry ? "active" : ""}`}
          onClick={onToggleTelemetry}
          title="Toggle telemetry overlay"
        >
          <IcTelemetry size={16}/>
        </button>
        <button className="top-icon" title="Keyboard shortcuts (planned)">
          <IcKeyboard size={16}/>
        </button>
        <button className="chal-chip" onClick={onChallengeClick} title="Change challenge">
          <span className="chal-chip-dot"/>
          <span className="chal-chip-name">
            {activeChallenge ? activeChallenge.name : "No challenge"}
          </span>
          <span className="chal-chip-kbd">C</span>
        </button>
      </div>
    </div>
  );
}

function Stat({ k, v, dim, accent }: { k: string; v: string; dim?: boolean; accent?: boolean }) {
  return (
    <div className="ts">
      <span className="ts-k">{k}</span>
      <span className={`ts-v ${dim ? "dim" : ""} ${accent ? "accent" : ""}`}>{v}</span>
    </div>
  );
}
