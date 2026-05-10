// Full-stage card overlay shown at gen 0 / step 0, before the user has
// pressed play. Disappears the moment the simulation advances.

import type { ChallengeSchema } from "../types";
import { IcPlay } from "./Icons";

interface Props {
  challenge: ChallengeSchema | null;
  population: number;
  sensorCount: number;
  actionCount: number;
  stepsPerGen: number;
  onPlay: () => void;
  onChooseChallenge: () => void;
}

export function EmptyState({
  challenge, population, sensorCount, actionCount, stepsPerGen,
  onPlay, onChooseChallenge,
}: Props) {
  return (
    <div className="empty">
      <div className="empty-card">
        <div className="empty-eyebrow"><span className="pulse"/>Generation 0 · ready</div>
        <h2 className="empty-title">A thousand random brains, no plan.</h2>
        <p className="empty-sub">
          Each agent carries a short genome wired into a tiny neural network.
          {challenge ? (
            <> Selection pressure is set to <strong style={{ color: "var(--text)" }}>{challenge.name}</strong> —{" "}
              {challenge.description.toLowerCase()} Press play and watch evolution find a strategy.</>
          ) : (
            <> No challenge is active — selection passes everyone, so nothing evolves. Choose one to give the population something to optimise.</>
          )}
        </p>
        <div className="empty-stats">
          <div><div className="empty-stat-k">Population</div><div className="empty-stat-v">{population.toLocaleString()}</div></div>
          <div><div className="empty-stat-k">Sensors / Actions</div><div className="empty-stat-v">{sensorCount} · {actionCount}</div></div>
          <div><div className="empty-stat-k">Steps / gen</div><div className="empty-stat-v">{stepsPerGen}</div></div>
        </div>
        <div className="empty-actions">
          <button className="empty-cta" onClick={onPlay}>
            <IcPlay size={14}/>
            <span>Run evolution</span>
            <span className="empty-cta-kbd">SPACE</span>
          </button>
          <button className="empty-secondary" onClick={onChooseChallenge}>
            Change challenge
          </button>
        </div>
      </div>
    </div>
  );
}
