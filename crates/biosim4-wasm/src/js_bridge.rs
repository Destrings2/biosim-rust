//! Bridge between JavaScript callbacks and biosim4-core's Sensor/Action traits.
//!
//! ## Threading
//!
//! `js_sys::Function` is `!Send + !Sync` because JS objects can't cross worker
//! boundaries. The Sensor/Action traits in biosim4-core require `Send + Sync`
//! (so `parallel` mode can run feed-forward across rayon threads). On wasm32
//! the simulator is single-threaded — we never actually share these across
//! threads — so we add `unsafe impl Send + Sync` wrappers. **Never call into
//! these wrappers from a non-main thread (e.g. from a Web Worker shared with
//! the main thread).**
//!
//! ## JS callback contracts
//!
//! ### Sensor callback: `(agent, simStep) => number`
//! - `agent`  — `AgentSnapshot` (id, x, y, heading, color, age, alive,
//!   breed_id, responsiveness, genome_length).
//! - `simStep` — current step index within the generation (0..steps_per_generation).
//! - Returns: a number in `[0.0, 1.0]`. Values outside the range are clamped.
//!   Throwing or returning non-numbers yields `0.0`.
//!
//! ### Action callback: `(agent, level) => effect | null`
//! - `agent` — same `AgentSnapshot` as above.
//! - `level` — raw neural activation (arbitrary float; `tanh` not pre-applied).
//! - Returns: `null` / `undefined` for no effect, or an `effect` object:
//!   ```js
//!   {
//!     dx: -1 | 0 | 1,         // optional — relative move on X
//!     dy: -1 | 0 | 1,         // optional — relative move on Y
//!     kill: true,             // optional — kill agent in front (requires kill_enable)
//!     emit_signal: 0,         // optional — layer index to mark with a pheromone
//!     responsiveness: 0..1,   // optional — overwrite this agent's responsiveness
//!   }
//!   ```

use biosim4_core::agent::AgentSnapshot;
use biosim4_core::registry::action::{Action, ActionContext};
use biosim4_core::registry::sensor::{Sensor, SensorContext};
use biosim4_core::types::Coord;
use js_sys::{Function, Reflect};
use wasm_bindgen::JsValue;

// ── JsSensor ──────────────────────────────────────────────────────────────

pub struct JsSensor {
    id: String,
    name: String,
    callback: Function,
}

// SAFETY: WASM is single-threaded. The biosim4-core Sensor trait requires
// Send + Sync to support the optional `parallel` feature; we never enable
// that feature when targeting wasm.
unsafe impl Send for JsSensor {}
unsafe impl Sync for JsSensor {}

impl JsSensor {
    pub fn new(id: String, name: String, callback: Function) -> Self {
        Self { id, name, callback }
    }
}

impl Sensor for JsSensor {
    fn id(&self) -> &str { &self.id }
    fn name(&self) -> &str { &self.name }

    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        let snapshot = AgentSnapshot::from_agent(ctx.agent);
        let snap_js = serde_wasm_bindgen::to_value(&snapshot).unwrap_or(JsValue::NULL);
        let step_js = JsValue::from_f64(ctx.sim_step as f64);

        match self.callback.call2(&JsValue::NULL, &snap_js, &step_js) {
            Ok(v) => v.as_f64().unwrap_or(0.0).clamp(0.0, 1.0) as f32,
            Err(_) => 0.0,
        }
    }
}

// ── JsAction ──────────────────────────────────────────────────────────────

pub struct JsAction {
    id: String,
    name: String,
    callback: Function,
}

unsafe impl Send for JsAction {}
unsafe impl Sync for JsAction {}

impl JsAction {
    pub fn new(id: String, name: String, callback: Function) -> Self {
        Self { id, name, callback }
    }
}

impl Action for JsAction {
    fn id(&self) -> &str { &self.id }
    fn name(&self) -> &str { &self.name }

    fn execute(&self, level: f32, ctx: &mut ActionContext) {
        let snapshot = AgentSnapshot::from_agent(ctx.agent);
        let snap_js = serde_wasm_bindgen::to_value(&snapshot).unwrap_or(JsValue::NULL);
        let level_js = JsValue::from_f64(level as f64);

        let result = match self.callback.call2(&JsValue::NULL, &snap_js, &level_js) {
            Ok(v) if !v.is_null() && !v.is_undefined() => v,
            _ => return,
        };

        // Try each known effect field. Unknown fields are silently ignored so
        // JS code is forward-compatible with new effect kinds.
        apply_effect(&result, ctx);
    }
}

fn get_field(obj: &JsValue, key: &str) -> Option<JsValue> {
    Reflect::get(obj, &JsValue::from_str(key)).ok().filter(|v| !v.is_undefined() && !v.is_null())
}

fn apply_effect(effect: &JsValue, ctx: &mut ActionContext) {
    // Movement: relative dx/dy in {-1, 0, 1}
    let dx = get_field(effect, "dx").and_then(|v| v.as_f64()).map(|f| f as i16).unwrap_or(0);
    let dy = get_field(effect, "dy").and_then(|v| v.as_f64()).map(|f| f as i16).unwrap_or(0);
    if dx != 0 || dy != 0 {
        let new_loc = Coord::new(ctx.agent.loc.x + dx.clamp(-1, 1), ctx.agent.loc.y + dy.clamp(-1, 1));
        if ctx.world.grid.is_in_bounds(new_loc) && ctx.world.grid.is_empty_at(new_loc) {
            ctx.move_queue.push((ctx.agent.id, new_loc));
        }
    }

    // Kill: queue death of agent directly in front of `last_move_dir`
    if get_field(effect, "kill").and_then(|v| v.as_bool()).unwrap_or(false)
        && ctx.config_kill_enable
    {
        let dir = ctx.agent.last_move_dir.as_normalized_coord();
        let target = Coord::new(ctx.agent.loc.x + dir.x, ctx.agent.loc.y + dir.y);
        if ctx.world.grid.is_occupied_at(target) {
            ctx.death_queue.push(ctx.world.grid.at(target));
        }
    }

    // Signal emission
    if let Some(layer_v) = get_field(effect, "emit_signal").and_then(|v| v.as_f64()) {
        let layer = layer_v as u8;
        if layer < ctx.signals.layer_count() {
            ctx.signals.increment(layer, ctx.agent.loc, ctx.world.grid);
        }
    }

    // Direct modulator overrides
    if let Some(r) = get_field(effect, "responsiveness").and_then(|v| v.as_f64()) {
        ctx.agent.responsiveness = (r as f32).clamp(0.0, 1.0);
    }
}
