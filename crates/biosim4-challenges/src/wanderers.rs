//! Smoke-test challenge for the programmable-agent abstraction.
//!
//! Spawns `count` non-evolved entities that wander randomly each step. They
//! don't kill peeps, don't compete for cells with any policy beyond grid
//! collision, and don't affect the survival selection (`evaluate` returns
//! "everyone passes"). They exist so the renderer, the `nearest_alien_dist`
//! sensor, and `ProgrammablePool::step_all` get exercised end-to-end before
//! more interesting challenges (predators, herders, …) plug into the same
//! abstraction.
//!
//! Pattern this challenge demonstrates for future challenges:
//!
//!  1. Define a `Program` struct (here `Wanderer`) that holds no per-entity
//!     state — anything per-entity goes in `this.state: [f32; 8]`.
//!  2. Inside `on_generation_start`, register the program via
//!     `register_or_get` and call `programmable.spawn(...)` for each
//!     entity you want in the world.
//!  3. Keep `evaluate` honest — return `(true, fitness)` for entities you
//!     want to pass, ignore programmables entirely (they're not peeps and
//!     are never evaluated as such).

use biosim4_core::agent::Agent;
use biosim4_core::programmable::{OwnerTag, Program, ProgramContext, ProgramOutput, Programmable};
use biosim4_core::registry::challenge::{Challenge, ChallengeOverlay, WorldMut};
use biosim4_core::types::Coord;
use biosim4_core::world::World;
use serde_json::{json, Value};

/// Owner tag for entities spawned by this challenge. Distinct integers are
/// just for `clear_for_owner` scoping when multiple challenges are active —
/// the value itself is meaningless beyond uniqueness.
const WANDERERS_TAG: OwnerTag = 0xA001;

/// The wandering behavior: pick a random adjacent cell each step, emit no
/// other effects. State slot 0 is the rng seed offset (unused for now;
/// reserved for future variants like "wander with drift").
struct Wanderer;

impl Program for Wanderer {
    fn id(&self) -> &str {
        "wanderer"
    }
    fn name(&self) -> &str {
        "Wanderer"
    }
    fn step(&self, this: &mut Programmable, ctx: &mut ProgramContext, out: &mut ProgramOutput) {
        // Pick a random of the 8 compass directions plus "stay put" (9 choices).
        // No fancy collision logic — the merge will silently drop the move if
        // the target cell is occupied, which gives a Brownian feel.
        let roll = ctx.rng.gen_range_u32(0, 9) as i16;
        let (dx, dy) = match roll {
            0 => (0, 0), // stay
            1 => (-1, 0),
            2 => (1, 0),
            3 => (0, -1),
            4 => (0, 1),
            5 => (-1, -1),
            6 => (1, -1),
            7 => (-1, 1),
            _ => (1, 1),
        };
        if dx == 0 && dy == 0 {
            return;
        }
        let dest = Coord::new(this.loc.x + dx, this.loc.y + dy);
        out.move_to = Some(dest);
        // Track heading so the renderer / debug overlays can show direction
        // even though we don't otherwise consult it.
        this.heading = Coord::new(dx, dy).as_dir();
    }
}

/// Demo challenge: spawn N wandering programmable entities at the start of
/// each generation. Survival is trivially "everyone passes" so this can be
/// composed with another challenge under `Composition::All` without
/// affecting selection (it's pure infrastructure exercise).
pub struct WanderersChallenge {
    pub count: u16,
    pub color: [u8; 3],
}

impl Default for WanderersChallenge {
    fn default() -> Self {
        // 8 wanderers in a bright cyan that contrasts against peeps and the
        // dark canvas. Small count so the smoke test stays cheap.
        Self { count: 8, color: [80, 220, 220] }
    }
}

impl Challenge for WanderersChallenge {
    fn id(&self) -> &str {
        "wanderers"
    }
    fn name(&self) -> &str {
        "Wanderers (demo)"
    }
    fn description(&self) -> &str {
        "Smoke-test challenge that spawns N non-evolved wandering entities each generation. \
         Doesn't filter survival — pair it with another challenge to see peeps adapt around it."
    }
    fn params_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "title": "count",
                    "minimum": 0,
                    "maximum": 256,
                    "default": 8
                },
                "color": {
                    "type": "array",
                    "title": "RGB color",
                    "items": { "type": "integer", "minimum": 0, "maximum": 255 },
                    "default": [80, 220, 220],
                    "minItems": 3,
                    "maxItems": 3
                }
            }
        })
    }
    fn configure(&mut self, params: Value) -> Result<(), String> {
        if let Some(c) = params.get("count").and_then(|v| v.as_u64()) {
            self.count = c.min(u16::MAX as u64) as u16;
        }
        if let Some(arr) = params.get("color").and_then(|v| v.as_array()) {
            if arr.len() == 3 {
                let mut rgb = [0u8; 3];
                for (i, v) in arr.iter().enumerate() {
                    rgb[i] = v.as_u64().unwrap_or(0).min(255) as u8;
                }
                self.color = rgb;
            }
        }
        Ok(())
    }

    fn on_generation_start(&mut self, ctx: &mut WorldMut) {
        // Register the program if missing, then spawn `count` wanderers at
        // random empty cells. The pool was already cleared inside
        // `reset_world` — we don't need to wipe again here.
        let prog = ctx.programmable.register_or_get("wanderer", || Box::new(Wanderer));
        for _ in 0..self.count {
            let loc = ctx.grid.find_empty_location(ctx.rng);
            // spawn returns None if the chosen cell raced (shouldn't happen
            // since we just found an empty one, but defensive).
            let _ = ctx.programmable.spawn(ctx.grid, prog, WANDERERS_TAG, loc, self.color);
        }
    }

    fn evaluate(&self, _agent: &Agent, _world: &World) -> (bool, f32) {
        // Pure passthrough. Compose with another challenge to add actual
        // selection pressure.
        (true, 1.0)
    }

    fn overlays(&self, _world: &World) -> Vec<ChallengeOverlay> {
        // Wanderers are visible on the grid already — no extra overlay.
        Vec::new()
    }
}
