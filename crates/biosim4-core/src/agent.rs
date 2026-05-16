//! Agent state struct.
//!
//! # Field groups
//!
//! - **Identity/lifecycle**: `id`, `alive`.
//! - **Spatial**: `loc`, `birth_loc`, `heading` (persistent — kept even when
//!   stationary), `last_move_dir`.
//! - **Visual**: `color` — derived from the first genome gene by
//!   `genome_color`. Minimum brightness is enforced (luminance ≥ 60) so
//!   agents don't disappear on the black background.
//! - **Life**: `age` — incremented each step in Phase 2 of `step_one_agent`.
//! - **Neural**: `genome`, `nnet` — the raw genome and its compiled network.
//! - **Neural modulators**: `responsiveness` (default 0.5), `osc_period`
//!   (default 34), `long_probe_dist` (default 16). These are written by the
//!   `set_responsiveness`, `set_oscillator_period`, and `set_longprobe_dist`
//!   actions at runtime.
//! - **Challenge tracking**: `challenge_bits` — 32-bit bitmask for challenges
//!   that require per-agent per-step state (e.g., `touch_any_wall`).
//! - **Extensibility**: `props` — a NetLogo-style property bag. `PropValue`
//!   supports f32, i32, bool, and String.
//!
//! `AgentSnapshot` is a lightweight serialization DTO for embedders that
//! need to ship per-agent state across an FFI / network boundary.

use crate::genome::{Genome, NeuralNet};
use crate::types::{Coord, Dir};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique identifier for a live agent. Corresponds to its slot index in [`Population`](crate::population::Population).
///
/// The value 0 is reserved as [`INVALID_AGENT`] and never assigned to a real agent.
pub type AgentId = u32;

/// Sentinel value meaning "no agent". Grid cells containing this value are empty.
pub const INVALID_AGENT: AgentId = 0;

/// Tagged property value for the agent's extensible [`Agent::props`] bag.
///
/// Custom sensors, actions, and challenges can store arbitrary per-agent
/// state here without modifying the `Agent` struct.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum PropValue {
    F32(f32),
    I32(i32),
    Bool(bool),
    Str(String),
}

/// A single simulated agent.
///
/// All fields are `pub` so sensors, actions, and challenge hooks can read and
/// write them directly without wrapper methods. See the module-level doc for
/// the logical groupings and lifecycle of each field group.
pub struct Agent {
    /// Unique stable identifier. Matches the agent's slot index in `Population`.
    pub id: AgentId,
    /// Whether this agent is still alive. Set to `false` by `drain_death_queue`.
    pub alive: bool,

    // Spatial state
    /// Current grid position.
    pub loc: Coord,
    /// Position at the start of the current generation.
    pub birth_loc: Coord,

    // Turtle-style properties
    /// Persistent heading — kept even when stationary.
    pub heading: Dir,
    /// RGB display color. Derived from the first gene of the genome at spawn.
    /// Minimum average luminance of 60 is enforced so agents remain visible
    /// against the black background.
    pub color: [u8; 3],

    // Life state
    /// Steps elapsed in the current generation. Incremented once per step in Phase 2.
    pub age: u32,
    /// Fractional energy in [0.0, 1.0]. Active only when `config.enable_energy` is true.
    pub energy: f32,

    // Neural state
    /// Raw genome: the sequence of [`Gene`](crate::genome::Gene) values that
    /// encode this agent's synaptic connections.
    pub genome: Genome,
    /// Compiled neural network. Rebuilt each generation from `genome` by
    /// [`create_wiring`](crate::genome::neural_net::create_wiring).
    pub nnet: NeuralNet,

    // Neural modulators — written by built-in actions, read by sensors and the step engine
    /// Sensitivity of neural response. Maps raw action levels through a sigmoid.
    /// Default: 0.5. Written by the `set_responsiveness` action.
    pub responsiveness: f32,
    /// Period of the oscillator sensor in steps. Default: 34.
    /// Written by the `set_oscillator_period` action.
    pub osc_period: u32,
    /// Probe distance for long-range sensors. Default: 16.
    /// Written by the `set_longprobe_dist` action.
    pub long_probe_dist: u32,

    /// Direction of the last successful move. Used by `last_move_dir_x/y` sensors.
    pub last_move_dir: Dir,

    /// Per-agent bitmask for challenges that track behavior during a generation.
    /// For example, `touch_any_wall` sets bit 0 when the agent reaches a border.
    pub challenge_bits: u32,

    /// Four persistent floating-point registers. Reset to 0.0 at the start of
    /// each generation. Written by `write_memory0..3` actions; read by
    /// `memory0..3` sensors.
    pub memory: [f32; 4],

    /// Extensible key-value property bag for custom extensions.
    /// Use [`get_prop`](Self::get_prop) and [`set_prop`](Self::set_prop) to access entries.
    pub props: HashMap<String, PropValue>,

    /// Per-individual mutation rate. The bit-flip operator reads this
    /// when `SimConfig.adaptive_mutation` is `true`; otherwise the
    /// global `cfg.point_mutation_rate` applies and this field is
    /// purely informational.
    pub mutation_rate: f32,
}

impl Agent {
    /// Create an agent with default modulator values and zero memory registers.
    ///
    /// `color` is derived from the first gene of `genome`. The agent starts
    /// `alive = true`, `age = 0`, and `energy = 1.0`.
    pub fn new(id: AgentId, loc: Coord, genome: Genome, nnet: NeuralNet) -> Self {
        let color = genome_color(&genome);
        Agent {
            id,
            alive: true,
            loc,
            birth_loc: loc,
            heading: Dir::default(),
            color,
            age: 0,
            energy: 1.0,
            genome,
            nnet,
            responsiveness: 0.5,
            osc_period: 34,
            long_probe_dist: 16,
            last_move_dir: Dir::default(),
            challenge_bits: 0,
            memory: [0.0; 4],
            props: HashMap::new(),
            // Sentinel; `spawn.rs` overwrites with the inherited (or
            // configured) rate after construction.
            mutation_rate: 0.0,
        }
    }

    /// Read a property from the extensible property bag by key.
    pub fn get_prop(&self, key: &str) -> Option<&PropValue> {
        self.props.get(key)
    }

    /// Write a property into the extensible property bag. Overwrites any existing value at `key`.
    pub fn set_prop(&mut self, key: &str, val: PropValue) {
        self.props.insert(key.to_string(), val);
    }
}

/// Derive a deterministic RGB color from the first bytes of the genome.
fn genome_color(genome: &Genome) -> [u8; 3] {
    if genome.is_empty() {
        return [128, 128, 128];
    }
    let raw = genome[0].0;
    let r = ((raw >> 16) & 0xFF) as u8;
    let g = ((raw >> 8) & 0xFF) as u8;
    let b = (raw & 0xFF) as u8;
    // Ensure minimum brightness (avoid black agents on black background)
    let lum = (r as u16 + g as u16 + b as u16) / 3;
    if lum < 60 {
        [r.saturating_add(80), g.saturating_add(80), b.saturating_add(80)]
    } else {
        [r, g, b]
    }
}

/// Serializable per-agent snapshot for FFI and network transport.
///
/// Contains only the fields needed for rendering and inspection. Construct
/// with [`AgentSnapshot::from_agent`]. The `heading` field stores the
/// [`Compass`](crate::types::dir::Compass) ordinal (0–8).
#[derive(Serialize, Deserialize)]
pub struct AgentSnapshot {
    /// Agent identifier (matches `Agent::id`).
    pub id: AgentId,
    /// Horizontal grid position.
    pub x: i16,
    /// Vertical grid position.
    pub y: i16,
    /// [`Compass`](crate::types::dir::Compass) ordinal of the agent's heading (0–8).
    pub heading: u8,
    /// RGB display color.
    pub color: [u8; 3],
    /// Steps elapsed in the current generation.
    pub age: u32,
    /// Whether the agent is alive.
    pub alive: bool,
    /// Current responsiveness modulator value.
    pub responsiveness: f32,
    /// Number of genes in the agent's genome.
    pub genome_length: usize,
}

impl AgentSnapshot {
    /// Build a snapshot from a live agent reference.
    pub fn from_agent(a: &Agent) -> Self {
        AgentSnapshot {
            id: a.id,
            x: a.loc.x,
            y: a.loc.y,
            heading: a.heading.0 as u8,
            color: a.color,
            age: a.age,
            alive: a.alive,
            responsiveness: a.responsiveness,
            genome_length: a.genome.len(),
        }
    }
}
