// Canvas renderer + main animation loop + tool dispatch.
//
// Click/drag behaviour depends on the active tool:
//   inspect   → click reports the agent id under the cursor (or null)
//   barrier   → click or drag paints barriers; clicking on an existing
//                barrier on mousedown switches into "erase" mode for the drag
//   kill      → click kills the agent under the cursor (no-op if empty)
//   reproduce → click spawns a mutated child of the agent under the cursor

import { useCallback, useEffect, useRef } from "react";
import type { Simulator } from "../../pkg/biosim4_wasm.js";
import type { EpochResult, SimStats } from "../types";
import type { Tool } from "./FloatingToolbar";

interface Props {
  simulator: Simulator;
  running: boolean;
  speed: number;
  pixelSize: number;
  onStats: (s: SimStats) => void;
  onEpoch: (e: EpochResult) => void;
  resetToken: number;
  /** The active tool; controls click/drag handling. */
  tool: Tool;
  /** For inspect tool: notified with agent id (or null) on click. */
  onAgentClick?: (agentId: number | null) => void;
  /** Called whenever the world is mutated (kill/reproduce/barrier) so the
   * parent can request a re-paint when the sim is paused. */
  onWorldChange?: () => void;
}

export function SimCanvas({
  simulator, running, speed, pixelSize, onStats, onEpoch, resetToken,
  tool, onAgentClick, onWorldChange,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number | null>(null);
  const runningRef = useRef(running);
  const speedRef = useRef(speed);
  const pixelSizeRef = useRef(pixelSize);
  const toolRef = useRef(tool);

  // For barrier-paint drag: whether we're currently dragging, and whether the
  // drag is in "erase" or "paint" mode (decided on mousedown by sampling the
  // cell under the cursor).
  const dragRef = useRef<{ erasing: boolean } | null>(null);
  // Avoid re-applying the same cell during a drag
  const lastCellRef = useRef<string | null>(null);

  useEffect(() => { runningRef.current = running; }, [running]);
  useEffect(() => { speedRef.current = speed; }, [speed]);
  useEffect(() => { pixelSizeRef.current = pixelSize; }, [pixelSize]);
  useEffect(() => { toolRef.current = tool; }, [tool]);

  // CSS pixel → world cell. The canvas backing buffer is sx×sy and is CSS-
  // scaled to whatever the parent gives it; `image-rendering: pixelated`
  // does the visual upscale. We compute the inverse from the live CSS size.
  const cssToWorld = useCallback((clientX: number, clientY: number) => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const sx = simulator.size_x();
    const sy = simulator.size_y();
    const x = Math.floor((clientX - rect.left) * sx / rect.width);
    const y = sy - 1 - Math.floor((clientY - rect.top) * sy / rect.height);
    return { x, y };
  }, [simulator]);

  const paint = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const sx = simulator.size_x();
    const sy = simulator.size_y();
    const frame = simulator.get_frame();
    const img = new ImageData(new Uint8ClampedArray(frame.length), sx, sy);
    img.data.set(frame);
    ctx.putImageData(img, 0, 0);

    try {
      const overlays = simulator.get_challenge_overlays() as any[];
      for (const ol of overlays) {
        ctx.fillStyle = `rgba(${ol.color[0]}, ${ol.color[1]}, ${ol.color[2]}, ${ol.color[3] / 255})`;
        if (ol.type === "circle") {
          ctx.beginPath();
          ctx.arc(ol.cx, sy - ol.cy, ol.radius, 0, 2 * Math.PI);
          ctx.fill();
        } else if (ol.type === "rectangle") {
          ctx.fillRect(ol.x, sy - ol.y - ol.h, ol.w, ol.h);
        } else if (ol.type === "points") {
          for (const pt of ol.points) {
            ctx.fillRect(pt[0] - ol.size/2, sy - pt[1] - ol.size/2, ol.size, ol.size);
          }
        }
      }
    } catch (err) {
      console.error("Failed to render overlays:", err);
    }
  }, [simulator]);

  // ── Mouse event wiring ──────────────────────────────────────────────
  // Single-click tools (inspect / kill / reproduce) handle on mousedown.
  // Barrier is a click-or-drag tool: we sample the cell on mousedown to
  // decide whether the drag paints or erases (so dragging from an existing
  // barrier always erases, dragging from empty space always paints).
  const onMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (e.button !== 0) return;
    const { x, y } = cssToWorld(e.clientX, e.clientY);
    const sx = simulator.size_x();
    const sy = simulator.size_y();
    if (x < 0 || y < 0 || x >= sx || y >= sy) return;

    switch (toolRef.current) {
      case "inspect": {
        const id = simulator.agent_at(x, y);
        onAgentClick?.(id > 0 ? id : null);
        return;
      }
      case "kill": {
        if (simulator.kill_at(x, y) > 0) {
          onWorldChange?.();
          paint();
        }
        return;
      }
      case "reproduce": {
        if (simulator.reproduce_at(x, y) > 0) {
          onWorldChange?.();
          paint();
        }
        return;
      }
      case "barrier": {
        const kind = simulator.cell_kind(x, y);
        if (kind === "agent" || kind === "oob") return;
        const erasing = kind === "barrier";
        dragRef.current = { erasing };
        lastCellRef.current = `${x},${y}`;
        simulator.set_barrier(x, y, !erasing);
        onWorldChange?.();
        paint();
        return;
      }
    }
  }, [cssToWorld, simulator, onAgentClick, onWorldChange, paint]);

  const onMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!dragRef.current || toolRef.current !== "barrier") return;
    const { x, y } = cssToWorld(e.clientX, e.clientY);
    const key = `${x},${y}`;
    if (key === lastCellRef.current) return;
    lastCellRef.current = key;
    if (simulator.cell_kind(x, y) === "agent") return;
    simulator.set_barrier(x, y, !dragRef.current.erasing);
    onWorldChange?.();
    paint();
  }, [cssToWorld, simulator, onWorldChange, paint]);

  const onMouseUp = useCallback(() => {
    dragRef.current = null;
    lastCellRef.current = null;
  }, []);

  // (re)start the animation loop when simulator instance or resetToken changes
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const sx = simulator.size_x();
    const sy = simulator.size_y();
    canvas.width = sx;
    canvas.height = sy;

    const offscreen = ctx.createImageData(sx, sy);

    const localPaint = () => {
      const frame = simulator.get_frame();
      offscreen.data.set(frame);
      ctx.putImageData(offscreen, 0, 0);

      try {
        const overlays = simulator.get_challenge_overlays() as any[];
        for (const ol of overlays) {
          ctx.fillStyle = `rgba(${ol.color[0]}, ${ol.color[1]}, ${ol.color[2]}, ${ol.color[3] / 255})`;
          if (ol.type === "circle") {
            ctx.beginPath();
            ctx.arc(ol.cx, sy - ol.cy, ol.radius, 0, 2 * Math.PI);
            ctx.fill();
          } else if (ol.type === "rectangle") {
            ctx.fillRect(ol.x, sy - ol.y - ol.h, ol.w, ol.h);
          } else if (ol.type === "points") {
            for (const pt of ol.points) {
              ctx.fillRect(pt[0] - ol.size/2, sy - pt[1] - ol.size/2, ol.size, ol.size);
            }
          }
        }
      } catch (err) {
        console.error("Failed to render overlays:", err);
      }
    };

    localPaint();

    const loop = () => {
      if (runningRef.current) {
        const stepsPerGen = simulator.steps_per_generation();
        let stepsTaken = 0;
        const maxPerFrame = Math.max(1, speedRef.current);
        while (stepsTaken < maxPerFrame) {
          if (simulator.sim_step() >= stepsPerGen) {
            try {
              const epoch = simulator.spawn_next_generation() as EpochResult;
              onEpoch(epoch);
            } catch (err) {
              console.error("spawn_next_generation failed:", err);
              runningRef.current = false;
              break;
            }
            break;
          }
          simulator.step();
          stepsTaken += 1;
        }
        localPaint();
        try { onStats(simulator.get_stats() as SimStats); } catch { /* ignore */ }
      }
      rafRef.current = requestAnimationFrame(loop);
    };

    rafRef.current = requestAnimationFrame(loop);

    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [simulator, resetToken]);

  const cursor =
    tool === "inspect"   ? "pointer"
  : tool === "barrier"   ? "crosshair"
  : tool === "kill"      ? "not-allowed"
  : tool === "reproduce" ? "copy"
  : "default";

  // pixelSize is honored for cursor calculation only — the canvas itself
  // fills its parent via CSS (`.canvas-frame canvas { width: 100% }`),
  // and `image-rendering: pixelated` upscales the world-grid bitmap.
  void pixelSize;

  return (
    <canvas
      ref={canvasRef}
      className="sim-canvas"
      style={{ cursor }}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseLeave={onMouseUp}
    />
  );
}
