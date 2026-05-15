//! Built-in action implementations (17 actions).
//!
//! # Conversion helpers
//!
//! Motor actions follow a four-step pipeline:
//!
//! 1. **Squash** the raw neural activation to a probability with
//!    [`level_to_prob`] (`|tanh|`) or [`level_to_signed_prob`]
//!    (`(tanh + 1)/2`, sign-preserving).
//! 2. **Scale** by the agent's responsiveness gate
//!    (`ctx.responsiveness_adjusted`, precomputed once per step from
//!    [`response_curve`]).
//! 3. **Threshold** (signal emission and kill require `> 0.5` before
//!    drawing).
//! 4. **Draw** a stochastic bool with [`prob2bool`], which now expects a
//!    pre-squashed probability in `[0, 1]`.
//!
//! [`prob2bool_responsive`] bundles steps 1, 2, and 4 for actions that
//! don't use a threshold (movement).
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

use biosim4_core::registry::{Action, ActionContext, ActionRegistry};
use biosim4_core::types::{Coord, Dir};

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
    registry.register(Box::new(EmitSignal1));
    registry.register(Box::new(EmitSignal2));
    registry.register(Box::new(MoveEast));
    registry.register(Box::new(MoveWest));
    registry.register(Box::new(MoveNorth));
    registry.register(Box::new(MoveSouth));
    registry.register(Box::new(MoveLeft));
    registry.register(Box::new(MoveRight));
    registry.register(Box::new(MoveReverse));
    registry.register(Box::new(KillForward));
    registry.register(Box::new(WriteMemory0));
    registry.register(Box::new(WriteMemory1));
    registry.register(Box::new(WriteMemory2));
    registry.register(Box::new(WriteMemory3));
}

// ── Utility functions ─────────────────────────────────────────────────────

/// Squash a raw neural activation level to a probability in `[0, 1]`. Used
/// as the input to [`prob2bool`] for actions that map a signed level to a
/// magnitude-only probability (movement, emission, kill).
#[inline]
pub fn level_to_prob(level: f32) -> f32 {
    level.tanh().abs()
}

/// Squash a raw neural activation level to `[0, 1]` via `(tanh + 1)/2`,
/// preserving sign information as a 0..0.5..1 mapping. Used by actions
/// that follow the "squash, scale by responsiveness, compare to a
/// threshold" pattern (signal emission, kill).
#[inline]
pub fn level_to_signed_prob(level: f32) -> f32 {
    (level.tanh() + 1.0) / 2.0
}

/// Draw a probabilistic bool from a pre-squashed probability. Caller is
/// responsible for ensuring `p ∈ [0, 1]` — debug builds assert it.
pub fn prob2bool(p: f32, rng: &mut biosim4_core::rng::Rng) -> bool {
    debug_assert!(
        (0.0..=1.0).contains(&p),
        "prob2bool expects a pre-squashed probability in [0, 1], got {p}"
    );
    rng.gen_bool(p)
}

/// Squash, scale by the agent's `responsiveness_adjusted`, and draw a
/// probabilistic bool. Used by every motor action whose probability
/// pipeline is `|tanh(level)| · responsivenessAdjusted`.
pub fn prob2bool_responsive(level: f32, ctx: &mut ActionContext) -> bool {
    let p = level_to_prob(level) * ctx.responsiveness_adjusted;
    ctx.rng.gen_bool(p)
}

/// Threshold pipeline used by signal emission and kill: sign-preserving
/// squash, responsiveness scale, gate on `> 0.5`, then probability draw.
/// Returns `true` if the action should fire this step.
pub fn fire_with_threshold(level: f32, threshold: f32, ctx: &mut ActionContext) -> bool {
    let p = level_to_signed_prob(level) * ctx.responsiveness_adjusted;
    p > threshold && prob2bool(p, ctx.rng)
}

/// Default `> 0.5` activation threshold shared by `EMIT_SIGNAL*` and
/// `KILL_FORWARD` — matches the reference simulator's midline gate.
pub const EMIT_THRESHOLD: f32 = 0.5;
pub const KILL_THRESHOLD: f32 = 0.5;

/// Lower bound of the action's "active range" — levels at or below this
/// remap to a zero firing probability.
pub const ACTION_MIN: f32 = 0.0;
/// Span of the action's "active range" — the level minus `ACTION_MIN` is
/// divided by this to produce the final firing probability. With the
/// defaults (`ACTION_MIN = 0`, `ACTION_RANGE = 1`) the remap is the
/// identity, leaving downstream tunability without affecting current
/// behavior.
pub const ACTION_RANGE: f32 = 1.0;

pub use biosim4_core::registry::action::response_curve;

/// Add a signed movement urge `level * dir.as_normalized_coord()` to the
/// per-agent X/Y accumulators. The accumulators are resolved into a single
/// grid step by `biosim4_core::registry::action::resolve_movement` after
/// every action has executed.
#[inline]
fn add_urge(ctx: &mut ActionContext, dir: Dir, level: f32) {
    let off = dir.as_normalized_coord();
    ctx.move_x_urge += off.x as f32 * level;
    ctx.move_y_urge += off.y as f32 * level;
}

// ── Internal state modulators ─────────────────────────────────────────────

// These three actions update internal agent state (responsiveness,
// oscillator period, long-probe distance) rather than producing motor
// output. They consume `level` directly and never multiply by
// `responsiveness_adjusted` — otherwise a low responsiveness would dampen
// the very signal an agent uses to raise itself out of that state, making
// it a one-way trap.

struct SetResponsiveness;
impl Action for SetResponsiveness {
    fn id(&self) -> &str {
        "set_responsiveness"
    }
    fn name(&self) -> &str {
        "set responsiveness"
    }
    fn execute(&self, level: f32, ctx: &mut ActionContext) {
        ctx.agent.responsiveness = ((level.tanh() + 1.0) / 2.0).clamp(0.0, 1.0);
    }
}

struct SetOscillatorPeriod;
impl Action for SetOscillatorPeriod {
    fn id(&self) -> &str {
        "set_oscillator_period"
    }
    fn name(&self) -> &str {
        "set oscillator period"
    }
    fn execute(&self, level: f32, ctx: &mut ActionContext) {
        let f01 = (level.tanh() + 1.0) / 2.0;
        ctx.agent.osc_period = (1.5 + (7.0 * f01).exp()) as u32 + 1;
    }
}

struct SetLongprobeDist;
impl Action for SetLongprobeDist {
    fn id(&self) -> &str {
        "set_longprobe_dist"
    }
    fn name(&self) -> &str {
        "set longprobe dist"
    }
    fn execute(&self, level: f32, ctx: &mut ActionContext) {
        let f01 = (level.tanh() + 1.0) / 2.0;
        ctx.agent.long_probe_dist = (1.0 + f01 * 32.0) as u32;
    }
}

// ── Signal emission ───────────────────────────────────────────────────────

struct EmitSignal0;
impl Action for EmitSignal0 {
    fn id(&self) -> &str {
        "emit_signal0"
    }
    fn name(&self) -> &str {
        "emit signal 0"
    }
    fn execute(&self, level: f32, ctx: &mut ActionContext) {
        if fire_with_threshold(level, EMIT_THRESHOLD, ctx) {
            ctx.signals.increment(0, ctx.agent.loc, ctx.world.grid);
        }
    }
}

struct EmitSignal1;
impl Action for EmitSignal1 {
    fn id(&self) -> &str {
        "emit_signal1"
    }
    fn name(&self) -> &str {
        "emit signal 1"
    }
    fn execute(&self, level: f32, ctx: &mut ActionContext) {
        if fire_with_threshold(level, EMIT_THRESHOLD, ctx) {
            ctx.signals.increment(1, ctx.agent.loc, ctx.world.grid);
        }
    }
}

struct EmitSignal2;
impl Action for EmitSignal2 {
    fn id(&self) -> &str {
        "emit_signal2"
    }
    fn name(&self) -> &str {
        "emit signal 2"
    }
    fn execute(&self, level: f32, ctx: &mut ActionContext) {
        if fire_with_threshold(level, EMIT_THRESHOLD, ctx) {
            ctx.signals.increment(2, ctx.agent.loc, ctx.world.grid);
        }
    }
}

// ── Kill ─────────────────────────────────────────────────────────────────

struct KillForward;
impl Action for KillForward {
    fn id(&self) -> &str {
        "kill_forward"
    }
    fn name(&self) -> &str {
        "kill forward"
    }
    fn execute(&self, level: f32, ctx: &mut ActionContext) {
        if !ctx.config_kill_enable {
            return;
        }
        // Same squash-and-scale as the signal-emit pipeline, then a kill-
        // specific remap of the probability through `ACTION_MIN` /
        // `ACTION_RANGE` before the draw.
        let p = level_to_signed_prob(level) * ctx.responsiveness_adjusted;
        if p <= KILL_THRESHOLD {
            return;
        }
        let p_kill = ((p - ACTION_MIN) / ACTION_RANGE).clamp(0.0, 1.0);
        if !prob2bool(p_kill, ctx.rng) {
            return;
        }
        let step = ctx.agent.last_move_dir.as_normalized_coord();
        let target = Coord::new(ctx.agent.loc.x + step.x, ctx.agent.loc.y + step.y);
        if ctx.world.grid.is_occupied_at(target) {
            let victim_id = ctx.world.grid.at(target);
            ctx.death_queue.push(victim_id);
        }
    }
}

// ── Movement contributions ────────────────────────────────────────────────
//
// Every movement action just adds a signed `level · direction` urge into
// the per-agent `move_x_urge` / `move_y_urge` accumulators on the context.
// `resolve_movement` runs once after the main dispatch loop and turns the
// final urge pair into at most one grid step, so urges that cancel along
// an axis (e.g. simultaneous `move_east` and `move_west`) collapse to no
// motion and orthogonal urges can combine into a single diagonal step.

macro_rules! simple_move {
    ($name:ident, $id:expr, $label:expr, $dir_expr:expr) => {
        struct $name;
        impl Action for $name {
            fn id(&self) -> &str {
                $id
            }
            fn name(&self) -> &str {
                $label
            }
            fn execute(&self, level: f32, ctx: &mut ActionContext) {
                let dir: Dir = $dir_expr(ctx);
                add_urge(ctx, dir, level);
            }
        }
    };
}

simple_move!(MoveEast, "move_east", "move east", |_ctx: &ActionContext| Dir(
    biosim4_core::types::Compass::E
));
simple_move!(MoveWest, "move_west", "move west", |_ctx: &ActionContext| Dir(
    biosim4_core::types::Compass::W
));
simple_move!(MoveNorth, "move_north", "move north", |_ctx: &ActionContext| Dir(
    biosim4_core::types::Compass::N
));
simple_move!(MoveSouth, "move_south", "move south", |_ctx: &ActionContext| Dir(
    biosim4_core::types::Compass::S
));
simple_move!(MoveForward, "move_forward", "move forward", |ctx: &ActionContext| ctx
    .agent
    .last_move_dir);
simple_move!(MoveReverse, "move_reverse", "move reverse", |ctx: &ActionContext| ctx
    .agent
    .last_move_dir
    .rotate180());
simple_move!(MoveLeft, "move_left", "move left", |ctx: &ActionContext| ctx
    .agent
    .last_move_dir
    .rotate90ccw());
simple_move!(MoveRight, "move_right", "move right", |ctx: &ActionContext| ctx
    .agent
    .last_move_dir
    .rotate90cw());

// MOVE_RL: same axis as MOVE_RIGHT; the sign of `level` chooses left vs
// right implicitly since `add_urge` multiplies the offset by `level`.
struct MoveRL;
impl Action for MoveRL {
    fn id(&self) -> &str {
        "move_rl"
    }
    fn name(&self) -> &str {
        "move RL"
    }
    fn execute(&self, level: f32, ctx: &mut ActionContext) {
        let dir = ctx.agent.last_move_dir.rotate90cw();
        add_urge(ctx, dir, level);
    }
}

// MOVE_RANDOM: draws a random8 direction once per step, then contributes
// `level · offset` into the urge sums.
struct MoveRandom;
impl Action for MoveRandom {
    fn id(&self) -> &str {
        "move_random"
    }
    fn name(&self) -> &str {
        "move random"
    }
    fn execute(&self, level: f32, ctx: &mut ActionContext) {
        let dir = Dir::random8(ctx.rng);
        add_urge(ctx, dir, level);
    }
}

// MOVE_X / MOVE_Y feed their raw level directly into a single axis.
struct MoveX;
impl Action for MoveX {
    fn id(&self) -> &str {
        "move_x"
    }
    fn name(&self) -> &str {
        "move X"
    }
    fn execute(&self, level: f32, ctx: &mut ActionContext) {
        ctx.move_x_urge += level;
    }
}

struct MoveY;
impl Action for MoveY {
    fn id(&self) -> &str {
        "move_y"
    }
    fn name(&self) -> &str {
        "move Y"
    }
    fn execute(&self, level: f32, ctx: &mut ActionContext) {
        ctx.move_y_urge += level;
    }
}

// ── Memory write actions ──────────────────────────────────────────────────

macro_rules! write_memory {
    ($name:ident, $id:literal, $label:literal, $reg:literal) => {
        struct $name;
        impl Action for $name {
            fn id(&self) -> &str {
                $id
            }
            fn name(&self) -> &str {
                $label
            }
            // Memory writes are internal-state updates, not motor outputs —
            // they consume `level` directly without the responsiveness gate
            // (otherwise low responsiveness would compress the writable range
            // toward 0.5 and break the agent's ability to flip stored bits).
            fn execute(&self, level: f32, ctx: &mut ActionContext) {
                ctx.agent.memory[$reg] = (level.tanh() + 1.0) / 2.0;
            }
        }
    };
}

write_memory!(WriteMemory0, "write_memory_0", "write memory 0", 0);
write_memory!(WriteMemory1, "write_memory_1", "write memory 1", 1);
write_memory!(WriteMemory2, "write_memory_2", "write memory 2", 2);
write_memory!(WriteMemory3, "write_memory_3", "write memory 3", 3);
