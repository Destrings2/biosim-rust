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
//! - **Breed/extensibility**: `breed_id`, `props` — a NetLogo-style property
//!   bag. `PropValue` supports f32, i32, bool, and String.
//!
//! `AgentSnapshot` is a lightweight serialization DTO for WASM/frontend use.

use std::collections::HashMap;
use crate::genome::{Genome, NeuralNet};
use crate::registry::breed::{BreedId, DEFAULT_BREED};
use crate::types::{Coord, Dir};
use serde::{Deserialize, Serialize};

pub type AgentId = u32;
pub const INVALID_AGENT: AgentId = 0;

/// Extensible property value — the turtle "variable" bag.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum PropValue {
    F32(f32),
    I32(i32),
    Bool(bool),
    Str(String),
}

/// A single simulated agent, inspired by NetLogo turtles.
pub struct Agent {
    pub id: AgentId,
    pub alive: bool,

    // Spatial state
    pub loc: Coord,
    pub birth_loc: Coord,

    // Turtle-style properties
    /// Persistent heading — kept even when stationary.
    pub heading: Dir,
    /// RGB color. Derived from genome by default; breed or props can override.
    pub color: [u8; 3],

    // Life state
    pub age: u32,
    pub energy: f32,

    // Neural state
    pub genome: Genome,
    pub nnet: NeuralNet,

    // Neural modulators
    pub responsiveness: f32,
    pub osc_period: u32,
    pub long_probe_dist: u32,

    // Backward-compat last-move tracking
    pub last_move_dir: Dir,

    // Challenge tracking
    pub challenge_bits: u32,

    // Memory registers — persist across steps, reset to 0 each generation
    pub memory: [f32; 4],

    // Breed and extensible properties
    pub breed_id: BreedId,
    pub props: HashMap<String, PropValue>,
}

impl Agent {
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
            breed_id: DEFAULT_BREED,
            props: HashMap::new(),
        }
    }

    pub fn get_prop(&self, key: &str) -> Option<&PropValue> { self.props.get(key) }
    pub fn set_prop(&mut self, key: &str, val: PropValue) { self.props.insert(key.to_string(), val); }
}

/// Derive a deterministic RGB color from the first bytes of the genome.
fn genome_color(genome: &Genome) -> [u8; 3] {
    if genome.is_empty() { return [128, 128, 128]; }
    let raw = genome[0].0;
    let r = ((raw >> 16) & 0xFF) as u8;
    let g = ((raw >> 8)  & 0xFF) as u8;
    let b = (raw & 0xFF) as u8;
    // Ensure minimum brightness (avoid black agents on black background)
    let lum = (r as u16 + g as u16 + b as u16) / 3;
    if lum < 60 {
        [r.saturating_add(80), g.saturating_add(80), b.saturating_add(80)]
    } else {
        [r, g, b]
    }
}

/// Lightweight snapshot for WASM/frontend serialization.
#[derive(Serialize, Deserialize)]
pub struct AgentSnapshot {
    pub id: AgentId,
    pub x: i16,
    pub y: i16,
    pub heading: u8,   // Compass ordinal
    pub color: [u8; 3],
    pub age: u32,
    pub alive: bool,
    pub breed_id: BreedId,
    pub responsiveness: f32,
    pub genome_length: usize,
}

impl AgentSnapshot {
    pub fn from_agent(a: &Agent) -> Self {
        AgentSnapshot {
            id: a.id,
            x: a.loc.x,
            y: a.loc.y,
            heading: a.heading.0 as u8,
            color: a.color,
            age: a.age,
            alive: a.alive,
            breed_id: a.breed_id,
            responsiveness: a.responsiveness,
            genome_length: a.genome.len(),
        }
    }
}
