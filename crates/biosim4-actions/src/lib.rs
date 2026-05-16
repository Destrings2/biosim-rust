//! Built-in action implementations (23 actions).
//!
//! Action structs live in themed submodules; this crate's public surface
//! is [`register_builtin_actions`] plus the conversion helpers re-exported
//! from [`util`] (so external action crates can reuse the same squash /
//! threshold pipeline).
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
//! **Interaction (4):** `emit_signal0..2` — deposit pheromone at the agent's
//! cell via `signals.increment`. `kill_forward` — queues death of the agent
//! directly ahead (only if `config.kill_enable`).
//!
//! **Memory writes (4):** `write_memory_0..3` — store a `(tanh + 1)/2`
//! mapped activation into the matching `agent.memory[i]` register.
//!
//! All movement actions push to `move_queue`; actual grid updates happen in
//! `drain_move_queue` at end-of-step.

mod kill;
mod memory;
mod modulators;
mod movement;
mod signals;
mod util;

pub use util::{
    fire_with_threshold, level_to_prob, level_to_signed_prob, prob2bool, prob2bool_responsive,
    response_curve, ACTION_MIN, ACTION_RANGE, EMIT_THRESHOLD, KILL_THRESHOLD,
};

use biosim4_core::registry::ActionRegistry;

pub fn register_builtin_actions(registry: &mut ActionRegistry) {
    registry.register(Box::new(movement::MoveX));
    registry.register(Box::new(movement::MoveY));
    registry.register(Box::new(movement::MoveForward));
    registry.register(Box::new(movement::MoveRL));
    registry.register(Box::new(movement::MoveRandom));
    registry.register(Box::new(modulators::SetOscillatorPeriod));
    registry.register(Box::new(modulators::SetLongprobeDist));
    registry.register(Box::new(modulators::SetResponsiveness));
    registry.register(Box::new(signals::EmitSignal0));
    registry.register(Box::new(signals::EmitSignal1));
    registry.register(Box::new(signals::EmitSignal2));
    registry.register(Box::new(movement::MoveEast));
    registry.register(Box::new(movement::MoveWest));
    registry.register(Box::new(movement::MoveNorth));
    registry.register(Box::new(movement::MoveSouth));
    registry.register(Box::new(movement::MoveLeft));
    registry.register(Box::new(movement::MoveRight));
    registry.register(Box::new(movement::MoveReverse));
    registry.register(Box::new(kill::KillForward));
    registry.register(Box::new(memory::WriteMemory0));
    registry.register(Box::new(memory::WriteMemory1));
    registry.register(Box::new(memory::WriteMemory2));
    registry.register(Box::new(memory::WriteMemory3));
}
