//! Built-in action implementations (17 actions).
//!
//! # Conversion helpers
//!
//! `prob2bool(level, rng)` converts a raw neural activation to a stochastic
//! boolean: `p = tanh(level).abs()`, then compare against a uniform random
//! draw. A level of 0.0 gives p=0 (never moves); large magnitude gives p≈1.
//!
//! `response_curve(r, k)` applies a non-linear transform that makes the agent's
//! `responsiveness` modulator affect how sharply action levels translate to
//! behavior. Default responsiveness is 0.5 (`k` defaults from config).
//!
//! # Action catalogue
//!
//! **Directional movement (8):** `move_east`, `move_west`, `move_north`,
//! `move_south` — unconditional cardinal moves (probabilistic via `prob2bool`).
//! `move_left`, `move_right` — relative to `heading`. `move_forward`,
//! `move_reverse` — along/against `heading`.
//!
//! **Composite movement (4):** `move_x`, `move_y` — axis-aligned probabilistic
//! moves (positive vs negative hemisphere). `move_rl` — left/right binary
//! split. `move_random` — uniform random among 8 directions.
//!
//! **Internal modulators (3):** `set_responsiveness`, `set_oscillator_period`,
//! `set_longprobe_dist` — update agent fields directly (not queued).
//!
//! **Interaction (2):** `emit_signal0` — deposits pheromone at agent location
//! via `signals.increment`. `kill_forward` — queues death of the agent
//! directly ahead (only if `config.kill_enable`).
//!
//! All movement actions push to `move_queue`; actual grid updates happen in
//! `drain_move_queue` at end-of-step.

use crate::registry::{Action, ActionContext, ActionRegistry};
use crate::types::{Coord, Dir};

pub fn register_builtin_actions(registry: &mut ActionRegistry) {
    registry.register(Box::new(MoveX));
    registry.register(Box::new(MoveY));
    registry.register(Box::new(MoveForward));
    registry.register(Box::new(MoveRL));
    registry.register(Box::new(MoveRandom));
    registry.register(Box::new(SetOscillatorPeriod));
    registry.register(Box::new(SetLongprobeDist));
    registry.register(Box::new(SetResponsiveness));
    registry.register(Box::new(EmitSignal0));
    registry.register(Box::new(MoveEast));
    registry.register(Box::new(MoveWest));
    registry.register(Box::new(MoveNorth));
    registry.register(Box::new(MoveSouth));
    registry.register(Box::new(MoveLeft));
    registry.register(Box::new(MoveRight));
    registry.register(Box::new(MoveReverse));
    registry.register(Box::new(KillForward));
}

// ── Utility functions ─────────────────────────────────────────────────────

/// Convert a raw activation level to a boolean probabilistically.
pub fn prob2bool(level: f32, rng: &mut crate::rng::Rng) -> bool {
    let p = level.tanh().abs();
    rng.gen_bool(p)
}

/// Non-linear responsiveness curve: dampens action reactivity.
pub fn response_curve(r: f32, k: f32) -> f32 {
    let r = r.clamp(0.0, 1.0);
    // k is the responsiveness curve factor (default 2.0)
    // Maps 0..1 → 0..1 with a non-linear shape
    (r - 2.0_f32.powf(-2.0 * k)).abs().powf(-2.0 * k).clamp(0.0, 1.0)
}

/// Try to queue a move in a given direction from current location.
fn try_move(ctx: &mut ActionContext, dir: Dir) {
    let step = dir.as_normalized_coord();
    let new_loc = Coord::new(ctx.agent.loc.x + step.x, ctx.agent.loc.y + step.y);
    if ctx.world.grid.is_in_bounds(new_loc) && ctx.world.grid.is_empty_at(new_loc) {
        ctx.move_queue.push((ctx.agent.id, new_loc));
    }
}



// ── Internal state modulators ─────────────────────────────────────────────

struct SetResponsiveness;
impl Action for SetResponsiveness {
    fn id(&self) -> &str { "set_responsiveness" }
    fn name(&self) -> &str { "set responsiveness" }
    fn execute(&self, level: f32, ctx: &mut ActionContext) {
        ctx.agent.responsiveness = ((level.tanh() + 1.0) / 2.0).clamp(0.0, 1.0);
    }
}

struct SetOscillatorPeriod;
impl Action for SetOscillatorPeriod {
    fn id(&self) -> &str { "set_oscillator_period" }
    fn name(&self) -> &str { "set oscillator period" }
    fn execute(&self, level: f32, ctx: &mut ActionContext) {
        let f01 = (level.tanh() + 1.0) / 2.0;
        ctx.agent.osc_period = (1.5 + (7.0 * f01).exp()) as u32 + 1;
    }
}

struct SetLongprobeDist;
impl Action for SetLongprobeDist {
    fn id(&self) -> &str { "set_longprobe_dist" }
    fn name(&self) -> &str { "set longprobe dist" }
    fn execute(&self, level: f32, ctx: &mut ActionContext) {
        let f01 = (level.tanh() + 1.0) / 2.0;
        ctx.agent.long_probe_dist = (1.0 + f01 * 32.0) as u32;
    }
}

// ── Signal emission ───────────────────────────────────────────────────────

struct EmitSignal0;
impl Action for EmitSignal0 {
    fn id(&self) -> &str { "emit_signal0" }
    fn name(&self) -> &str { "emit signal 0" }
    fn execute(&self, level: f32, ctx: &mut ActionContext) {
        if prob2bool(level, ctx.rng) {
            ctx.signals.increment(0, ctx.agent.loc, ctx.world.grid);
        }
    }
}

// ── Kill ─────────────────────────────────────────────────────────────────

struct KillForward;
impl Action for KillForward {
    fn id(&self) -> &str { "kill_forward" }
    fn name(&self) -> &str { "kill forward" }
    fn execute(&self, level: f32, ctx: &mut ActionContext) {
        if !ctx.config_kill_enable { return; }
        if !prob2bool(level, ctx.rng) { return; }
        let step = ctx.agent.last_move_dir.as_normalized_coord();
        let target = Coord::new(ctx.agent.loc.x + step.x, ctx.agent.loc.y + step.y);
        if ctx.world.grid.is_occupied_at(target) {
            let victim_id = ctx.world.grid.at(target);
            ctx.death_queue.push(victim_id);
        }
    }
}

// ── Individual directional moves ─────────────────────────────────────────

macro_rules! simple_move {
    ($name:ident, $id:expr, $label:expr, $dir_expr:expr) => {
        struct $name;
        impl Action for $name {
            fn id(&self) -> &str { $id }
            fn name(&self) -> &str { $label }
            fn execute(&self, level: f32, ctx: &mut ActionContext) {
                let dir: Dir = $dir_expr(ctx);
                if prob2bool(level, ctx.rng) { try_move(ctx, dir); }
            }
        }
    };
}

simple_move!(MoveEast,    "move_east",    "move east",    |_ctx: &ActionContext| Dir(crate::types::Compass::E));
simple_move!(MoveWest,    "move_west",    "move west",    |_ctx: &ActionContext| Dir(crate::types::Compass::W));
simple_move!(MoveNorth,   "move_north",   "move north",   |_ctx: &ActionContext| Dir(crate::types::Compass::N));
simple_move!(MoveSouth,   "move_south",   "move south",   |_ctx: &ActionContext| Dir(crate::types::Compass::S));
simple_move!(MoveForward, "move_forward", "move forward", |ctx: &ActionContext| ctx.agent.last_move_dir);
simple_move!(MoveReverse, "move_reverse", "move reverse", |ctx: &ActionContext| ctx.agent.last_move_dir.rotate180());
simple_move!(MoveLeft,    "move_left",    "move left",    |ctx: &ActionContext| ctx.agent.last_move_dir.rotate90ccw());
simple_move!(MoveRight,   "move_right",   "move right",   |ctx: &ActionContext| ctx.agent.last_move_dir.rotate90cw());

struct MoveRL;
impl Action for MoveRL {
    fn id(&self) -> &str { "move_rl" }
    fn name(&self) -> &str { "move RL" }
    fn execute(&self, level: f32, ctx: &mut ActionContext) {
        let dir = if level >= 0.0 {
            ctx.agent.last_move_dir.rotate90cw()
        } else {
            ctx.agent.last_move_dir.rotate90ccw()
        };
        if prob2bool(level, ctx.rng) { try_move(ctx, dir); }
    }
}

struct MoveRandom;
impl Action for MoveRandom {
    fn id(&self) -> &str { "move_random" }
    fn name(&self) -> &str { "move random" }
    fn execute(&self, level: f32, ctx: &mut ActionContext) {
        if prob2bool(level, ctx.rng) {
            let dir = Dir::random8(ctx.rng);
            try_move(ctx, dir);
        }
    }
}

/// Accumulate X and Y components from several move actions into a combined move.
/// This mirrors the C++ combined-move logic.
struct MoveX;
impl Action for MoveX {
    fn id(&self) -> &str { "move_x" }
    fn name(&self) -> &str { "move X" }
    fn execute(&self, level: f32, ctx: &mut ActionContext) {
        // Standalone MoveX: treat as pure east/west
        let v = level.tanh();
        let dir = if v >= 0.0 { Dir(crate::types::Compass::E) } else { Dir(crate::types::Compass::W) };
        if ctx.rng.gen_bool(v.abs()) { try_move(ctx, dir); }
    }
}

struct MoveY;
impl Action for MoveY {
    fn id(&self) -> &str { "move_y" }
    fn name(&self) -> &str { "move Y" }
    fn execute(&self, level: f32, ctx: &mut ActionContext) {
        let v = level.tanh();
        let dir = if v >= 0.0 { Dir(crate::types::Compass::N) } else { Dir(crate::types::Compass::S) };
        if ctx.rng.gen_bool(v.abs()) { try_move(ctx, dir); }
    }
}
