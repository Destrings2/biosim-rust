// Default boilerplate shown the first time the user opens the custom-challenge
// editor. Survival = stay inside a centred circle of configurable radius.

export const BOILERPLATE = `/// <reference path="biosim-api.d.ts" />
// Custom challenge. Edit then click "Save & Apply".
// Hover any identifier for type docs; \`agent.\` and \`world.\` autocomplete.

({
  id: "custom_circle",
  name: "Custom circle",
  description: "Stay inside a centred circle of radius * min(w, h).",

  paramsSchema: {
    type: "object",
    properties: {
      radius: { type: "number", minimum: 0.05, maximum: 0.5, default: 0.25,
                description: "Fraction of the shorter world axis." },
    },
  },

  configure(params) {
    this.radius = typeof params.radius === "number" ? params.radius : 0.25;
  },

  evaluate(agent, world) {
    const r = (this.radius ?? 0.25) * Math.min(world.size_x, world.size_y);
    const cx = world.size_x / 2;
    const cy = world.size_y / 2;
    const d = Math.hypot(agent.x - cx, agent.y - cy);
    return { pass: d <= r, fitness: Math.max(0, 1 - d / r) };
  },

  overlays(world) {
    // Overlay coordinates are in world cells (0..size_x, 0..size_y).
    const r = (this.radius ?? 0.25) * Math.min(world.size_x, world.size_y);
    return [{
      type: "circle",
      cx: world.size_x / 2, cy: world.size_y / 2, radius: r,
      color: [255, 200, 0, 80],
    }];
  },
})
`;
