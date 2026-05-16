//! Movement actions.
//!
//! Every movement action just adds a signed `level · direction` urge into
//! the per-agent `move_x_urge` / `move_y_urge` accumulators on the
//! context. `resolve_movement` runs once after the main dispatch loop and
//! turns the final urge pair into at most one grid step, so urges that
//! cancel along an axis (e.g. simultaneous `move_east` and `move_west`)
//! collapse to no motion and orthogonal urges can combine into a single
//! diagonal step.

use crate::util::add_urge;
use biosim4_core::registry::{Action, ActionContext};
use biosim4_core::types::Dir;

macro_rules! simple_move {
    ($name:ident, $id:expr, $label:expr, $dir_expr:expr) => {
        pub(crate) struct $name;
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

/// MOVE_RL: same axis as MOVE_RIGHT; the sign of `level` chooses left vs
/// right implicitly since `add_urge` multiplies the offset by `level`.
pub(crate) struct MoveRL;
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

/// MOVE_RANDOM: draws a random8 direction once per step, then contributes
/// `level · offset` into the urge sums.
pub(crate) struct MoveRandom;
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

/// MOVE_X feeds its raw level directly into the X axis accumulator.
pub(crate) struct MoveX;
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

/// MOVE_Y feeds its raw level directly into the Y axis accumulator.
pub(crate) struct MoveY;
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
