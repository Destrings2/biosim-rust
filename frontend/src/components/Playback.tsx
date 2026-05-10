// Playback bar pinned under the canvas: play/pause, step, gen, epoch, reset
// + speed and pixel sliders.

import { useEffect, useState } from "react";
import { IcEpoch, IcPause, IcPlay, IcReset, IcStep, IcStepGen } from "./Icons";

interface Props {
  paused: boolean;
  onTogglePlay: () => void;
  onStep: () => void;
  onStepGen: () => void;
  onRunEpoch: () => void;
  onReset: () => void;

  speed: number;
  onSpeedChange: (v: number) => void;
  pixel: number;
  onPixelChange: (v: number) => void;
}

export function Playback({
  paused, onTogglePlay, onStep, onStepGen, onRunEpoch, onReset,
  speed, onSpeedChange, pixel, onPixelChange,
}: Props) {
  // PX slider: changing the value resizes the canvas, which causes a visible
  // "flicker" if it fires every frame while dragging. We track a local
  // preview value and only commit to the parent on pointer/key release.
  const [previewPx, setPreviewPx] = useState(pixel);
  useEffect(() => { setPreviewPx(pixel); }, [pixel]);
  const commit = () => { if (previewPx !== pixel) onPixelChange(previewPx); };

  return (
    <div className="playback">
      <div className="pb-group">
        <button className="pb-btn primary" onClick={onTogglePlay} title="Play / Pause (Space)">
          {paused ? <IcPlay size={12}/> : <IcPause size={12}/>}
          <span>{paused ? "PLAY" : "PAUSE"}</span>
        </button>
        <button className="pb-btn" onClick={onStep} disabled={!paused} title="Step ×1">
          <IcStep size={13}/><span>STEP</span>
        </button>
        <button className="pb-btn" onClick={onStepGen} disabled={!paused} title="Step generation">
          <IcStepGen size={13}/><span>GEN</span>
        </button>
        <button className="pb-btn" onClick={onRunEpoch} disabled={!paused} title="Run epoch">
          <IcEpoch size={13}/><span>EPOCH</span>
        </button>
        <button className="pb-btn danger" onClick={onReset} title="Reset to gen 0">
          <IcReset size={13}/><span>RESET</span>
        </button>
      </div>
      <div className="pb-group pb-slider-group">
        <div className="pb-slider">
          <span>SPF</span>
          <input type="range" min="1" max="64" value={speed}
                 onChange={(e) => onSpeedChange(+e.target.value)} />
          <span style={{ color: "var(--text)", minWidth: 24 }}>{speed}×</span>
        </div>
        <div className="pb-slider">
          <span>PX</span>
          <input
            type="range" min="1" max="8" value={previewPx}
            onChange={(e) => setPreviewPx(+e.target.value)}
            onMouseUp={commit}
            onTouchEnd={commit}
            onKeyUp={commit}
            onBlur={commit}
          />
          <span style={{ color: "var(--text)", minWidth: 24 }}>{previewPx}×</span>
        </div>
      </div>
    </div>
  );
}
