//! Squash / threshold helpers shared by every motor action.
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
//! 4. **Draw** a stochastic bool with [`prob2bool`], which expects a
//!    pre-squashed probability in `[0, 1]`.
//!
//! [`prob2bool_responsive`] bundles steps 1, 2, and 4 for actions that
//! don't use a threshold (movement).

use biosim4_core::registry::ActionContext;
use biosim4_core::types::Dir;

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
pub(crate) fn add_urge(ctx: &mut ActionContext, dir: Dir, level: f32) {
    let off = dir.as_normalized_coord();
    ctx.move_x_urge += off.x as f32 * level;
    ctx.move_y_urge += off.y as f32 * level;
}
