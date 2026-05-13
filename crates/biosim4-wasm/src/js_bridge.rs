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

use biosim4_core::agent::{Agent, AgentSnapshot};
use biosim4_core::registry::action::{Action, ActionContext};
use biosim4_core::registry::challenge::{Challenge, ChallengeOverlay, WorldMut};
use biosim4_core::registry::sensor::{Sensor, SensorContext};
use biosim4_core::types::Coord;
use biosim4_core::world::World;
use js_sys::{Function, Object, Reflect};
use serde::Serialize;
use serde_json::Value;
use wasm_bindgen::{JsCast, JsValue};

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

// ── JsChallenge ───────────────────────────────────────────────────────────
//
// A JavaScript-defined challenge. The user passes an object literal with
// fields:
//   id, name, description       — strings
//   paramsSchema                — JSON schema object
//   configure(params)           — optional, mutates `this`
//   evaluate(agent, world)      — required, returns { pass, fitness }
//   onSimStep(ctx) / onGenerationStart(ctx) — optional
//   overlays(world)             — optional, returns ChallengeOverlay[]
//
// We hold the object as `this` and pull cached `Function` handles for fast
// per-call dispatch.

pub struct JsChallenge {
    id: String,
    name: String,
    description: String,
    params_schema: Value,
    this: Object,
    eval_fn: Function,
    on_step_fn: Option<Function>,
    on_gen_fn: Option<Function>,
    overlays_fn: Option<Function>,
    configure_fn: Option<Function>,
}

unsafe impl Send for JsChallenge {}
unsafe impl Sync for JsChallenge {}

#[derive(Serialize)]
struct WorldView {
    size_x: u16,
    size_y: u16,
    steps_per_generation: u32,
    generation: u32,
    step: u32,
}

impl<'a> From<&World<'a>> for WorldView {
    fn from(w: &World<'a>) -> Self {
        Self {
            size_x: w.size_x,
            size_y: w.size_y,
            steps_per_generation: w.steps_per_generation,
            generation: w.generation,
            step: w.step,
        }
    }
}

#[derive(Serialize)]
struct StepCtxView {
    size_x: u16,
    size_y: u16,
    generation: u32,
    step: u32,
}

impl JsChallenge {
    /// Build a JsChallenge from a JS object. Returns `Err(String)` if required
    /// fields are missing or malformed.
    pub fn from_object(obj: JsValue) -> Result<Self, String> {
        if !obj.is_object() {
            return Err("Challenge must be an object literal".to_string());
        }
        let obj: Object = obj.dyn_into_object()?;

        let id = read_string(&obj, "id").ok_or_else(|| "challenge.id (string) is required".to_string())?;
        if id.is_empty() {
            return Err("challenge.id must be a non-empty string".to_string());
        }
        let name = read_string(&obj, "name").unwrap_or_else(|| id.clone());
        let description = read_string(&obj, "description").unwrap_or_default();

        let eval_fn = read_function(&obj, "evaluate")
            .ok_or_else(|| "challenge.evaluate(agent, world) is required".to_string())?;

        let on_step_fn = read_function(&obj, "onSimStep");
        let on_gen_fn = read_function(&obj, "onGenerationStart");
        let overlays_fn = read_function(&obj, "overlays");
        let configure_fn = read_function(&obj, "configure");

        // paramsSchema (optional). Defaults to an empty object schema.
        let params_schema = match Reflect::get(&obj, &JsValue::from_str("paramsSchema")) {
            Ok(v) if !v.is_undefined() && !v.is_null() => {
                serde_wasm_bindgen::from_value::<Value>(v)
                    .unwrap_or_else(|_| serde_json::json!({ "type": "object", "properties": {} }))
            }
            _ => serde_json::json!({ "type": "object", "properties": {} }),
        };

        Ok(Self {
            id, name, description, params_schema,
            this: obj, eval_fn,
            on_step_fn, on_gen_fn, overlays_fn, configure_fn,
        })
    }
}

impl Challenge for JsChallenge {
    fn id(&self) -> &str { &self.id }
    fn name(&self) -> &str { &self.name }
    fn description(&self) -> &str { &self.description }

    fn params_schema(&self) -> Value {
        self.params_schema.clone()
    }

    fn configure(&mut self, params: Value) -> Result<(), String> {
        let Some(f) = &self.configure_fn else { return Ok(()) };
        let params_js = serde_wasm_bindgen::to_value(&params)
            .map_err(|e| format!("configure: failed to marshal params: {e}"))?;
        f.call1(&self.this, &params_js)
            .map(|_| ())
            .map_err(|e| format!("configure threw: {}", js_err_to_string(&e)))
    }

    fn evaluate(&self, agent: &Agent, world: &World) -> (bool, f32) {
        let snap = AgentSnapshot::from_agent(agent);
        let snap_js = serde_wasm_bindgen::to_value(&snap).unwrap_or(JsValue::NULL);
        let world_view: WorldView = world.into();
        let world_js = serde_wasm_bindgen::to_value(&world_view).unwrap_or(JsValue::NULL);

        let result = match self.eval_fn.call2(&self.this, &snap_js, &world_js) {
            Ok(v) => v,
            Err(_) => return (false, 0.0),
        };
        // result: { pass: bool, fitness: number }  — or a bare number/bool.
        if let Some(n) = result.as_f64() {
            let s = (n as f32).clamp(0.0, 1.0);
            return (s >= 1.0, s);
        }
        if let Some(b) = result.as_bool() {
            return (b, if b { 1.0 } else { 0.0 });
        }
        let pass = get_field(&result, "pass").and_then(|v| v.as_bool()).unwrap_or(false);
        let fitness = get_field(&result, "fitness").and_then(|v| v.as_f64())
            .map(|f| (f as f32).clamp(0.0, 1.0))
            .unwrap_or(0.0);
        (pass, fitness)
    }

    fn on_sim_step(&mut self, ctx: &mut WorldMut) {
        let Some(f) = &self.on_step_fn else { return };
        let view = StepCtxView {
            size_x: ctx.grid.size_x,
            size_y: ctx.grid.size_y,
            generation: ctx.generation,
            step: ctx.step,
        };
        let view_js = serde_wasm_bindgen::to_value(&view).unwrap_or(JsValue::NULL);
        let _ = f.call1(&self.this, &view_js);
        // Side effects (queued deaths, signal emissions) are not yet plumbed
        // through here; the JS callback can still mutate `this` to maintain
        // per-step state, which is sufficient for most challenges.
    }

    fn on_generation_start(&mut self, ctx: &mut WorldMut) {
        let Some(f) = &self.on_gen_fn else { return };
        let view = StepCtxView {
            size_x: ctx.grid.size_x,
            size_y: ctx.grid.size_y,
            generation: ctx.generation,
            step: ctx.step,
        };
        let view_js = serde_wasm_bindgen::to_value(&view).unwrap_or(JsValue::NULL);
        let _ = f.call1(&self.this, &view_js);
    }

    fn overlays(&self, world: &World) -> Vec<ChallengeOverlay> {
        let Some(f) = &self.overlays_fn else { return Vec::new() };
        let world_view: WorldView = world.into();
        let world_js = serde_wasm_bindgen::to_value(&world_view).unwrap_or(JsValue::NULL);
        let result = match f.call1(&self.this, &world_js) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };
        serde_wasm_bindgen::from_value::<Vec<ChallengeOverlay>>(result).unwrap_or_default()
    }
}

// ── helpers ───────────────────────────────────────────────────────────────

fn read_string(obj: &Object, key: &str) -> Option<String> {
    Reflect::get(obj, &JsValue::from_str(key)).ok().and_then(|v| v.as_string())
}

fn read_function(obj: &Object, key: &str) -> Option<Function> {
    let v = Reflect::get(obj, &JsValue::from_str(key)).ok()?;
    if v.is_function() { v.dyn_into::<Function>().ok() } else { None }
}

fn js_err_to_string(e: &JsValue) -> String {
    e.as_string()
        .or_else(|| Reflect::get(e, &JsValue::from_str("message")).ok().and_then(|m| m.as_string()))
        .unwrap_or_else(|| "unknown JS error".to_string())
}

// Tiny convenience to convert a JsValue into Object with a string error.
trait JsValueExt {
    fn dyn_into_object(self) -> Result<Object, String>;
}
impl JsValueExt for JsValue {
    fn dyn_into_object(self) -> Result<Object, String> {
        self.dyn_into::<Object>().map_err(|_| "expected a plain object".to_string())
    }
}
