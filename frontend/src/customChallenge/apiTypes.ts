// Ambient declarations for the custom-challenge JavaScript API.
//
// Exported as a string and fed to Monaco via `addExtraLib(...)` so the editor
// offers full autocomplete and hover docs while the user writes a challenge.

export const API_DTS = `
/** Per-agent snapshot passed to \`evaluate\`. Coordinates are world-integer. */
declare interface Agent {
  /** Stable agent id. Non-zero. */
  readonly id: number;
  /** World X cell (0..world.size_x-1). */
  readonly x: number;
  /** World Y cell (0..world.size_y-1). */
  readonly y: number;
  /** Compass ordinal 0..7 (N=0, NE=1, …). */
  readonly heading: number;
  /** RGB triple. */
  readonly color: [number, number, number];
  /** Steps lived this generation. */
  readonly age: number;
  /** False if the agent died before evaluation. */
  readonly alive: boolean;
  /** Numeric breed id (default 0). */
  readonly breed_id: number;
  /** Current responsiveness modulator (0..1). */
  readonly responsiveness: number;
  /** Length of the agent's genome in genes. */
  readonly genome_length: number;
}

/** Read-only world view passed to \`evaluate\` and \`overlays\`. */
declare interface WorldView {
  /** Grid width in cells. */
  readonly size_x: number;
  /** Grid height in cells. */
  readonly size_y: number;
  /** Generation length. */
  readonly steps_per_generation: number;
  /** Current generation index (0-based). */
  readonly generation: number;
  /** Current step within the generation. */
  readonly step: number;
}

/** Smaller view passed to step hooks. */
declare interface StepContext {
  readonly size_x: number;
  readonly size_y: number;
  readonly generation: number;
  readonly step: number;
}

/** Return shape of \`evaluate\`. \`pass\` decides survival; \`fitness\` (0..1) ranks. */
declare interface ChallengeResult {
  pass: boolean;
  fitness: number;
}

/** Visual overlays rendered above the world canvas. Coordinates are in world
 *  cell space (\`0..world.size_x\`, \`0..world.size_y\`); the renderer maps cells
 *  to canvas pixels 1:1. Color is RGBA bytes 0..255. */
declare type ChallengeOverlay =
  | { type: "circle"; cx: number; cy: number; radius: number; color: [number, number, number, number] }
  | { type: "rectangle"; x: number; y: number; w: number; h: number; color: [number, number, number, number] }
  | { type: "points"; points: Array<[number, number]>; color: [number, number, number, number]; size: number };

/**
 * JSON Schema (draft-07 object) describing the challenge's configurable
 * parameters. The frontend renders a form from this; \`configure(params)\`
 * receives the filled-in values.
 */
declare interface ParamsSchema {
  type: "object";
  properties?: Record<string, {
    type: "number" | "boolean" | "string";
    minimum?: number;
    maximum?: number;
    default?: number | boolean | string;
    description?: string;
  }>;
}

/**
 * A user-defined challenge. Implement \`evaluate\` at minimum. All other methods
 * are optional. \`this\` is the challenge object itself — store any per-instance
 * state (radius, accumulators, etc.) on it.
 */
declare interface Challenge {
  /** Stable identifier. Must be a non-empty string. */
  id: string;
  /** Human-friendly name shown in the picker. */
  name: string;
  /** One-line description shown under the name. */
  description?: string;
  /** JSON Schema for configurable parameters (optional). */
  paramsSchema?: ParamsSchema;
  /** Apply params to \`this\`. Called when the user clicks Apply. */
  configure?(params: Record<string, unknown>): void;
  /** Decide pass/fitness for a single agent. Required. */
  evaluate(agent: Agent, world: WorldView): ChallengeResult | boolean | number;
  /** Optional per-step hook (read-only). */
  onSimStep?(ctx: StepContext): void;
  /** Optional generation-start hook. */
  onGenerationStart?(ctx: StepContext): void;
  /** Visual overlays to render over the canvas. */
  overlays?(world: WorldView): ChallengeOverlay[];
}
`;
