//! Integration tests for the public `Simulator` API.
//!
//! These tests exercise the wasm-bindgen surface, so they must run under a
//! wasm runtime (`wasm-pack test --node` or `wasm-pack test --headless`).
//! On native targets the entire file is empty — `#[wasm_bindgen]` methods
//! call into wasm-bindgen runtime stubs that panic on non-wasm32 targets.
//!
//! ```bash
//! cargo install wasm-pack
//! wasm-pack test --node crates/biosim4-wasm
//! ```
//!
//! The pure-Rust render logic is covered by unit tests in `src/render.rs`,
//! which run natively under `cargo test`.

#![cfg(target_arch = "wasm32")]

use biosim4_wasm::Simulator;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

fn small_config_json() -> String {
    serde_json::to_string(&serde_json::json!({
        "size_x": 16,
        "size_y": 16,
        "population": 10,
        "num_threads": 1,
        "deterministic": true,
        "rng_seed": 7,
        "signal_layers": 1,
        "steps_per_generation": 8,
        "max_generations": 5,
        "genome_initial_length_min": 8,
        "genome_initial_length_max": 8,
        "genome_max_length": 50,
        "max_number_neurons": 3,
        "point_mutation_rate": 0.001,
        "gene_insertion_deletion_rate": 0.0,
        "deletion_ratio": 0.5,
        "sexual_reproduction": false,
        "choose_parents_by_fitness": false,
        "kill_enable": false,
        "responsiveness": 0.5,
        "responsiveness_curve_k_factor": 2.0,
        "population_sensor_radius": 2.5,
        "signal_sensor_radius": 2.0,
        "long_probe_distance": 16,
        "short_probe_barrier_distance": 4,
        "barrier_type": 0,
        "genome_analysis_stride": 25,
        "display_sample_genomes": 3,
        "genome_comparison_method": 0,
        "save_video": false,
        "video_stride": 25
    })).unwrap()
}

#[wasm_bindgen_test]
fn simulator_constructs_with_default_config() {
    let sim = Simulator::new("").expect("default config should construct");
    assert_eq!(sim.generation(), 0);
    assert_eq!(sim.sim_step(), 0);
    assert_eq!(sim.steps_per_generation(), 300);
    assert!(sim.alive_count() > 0);
}

#[wasm_bindgen_test]
fn simulator_constructs_with_custom_config() {
    let sim = Simulator::new(&small_config_json()).expect("config should parse");
    assert_eq!(sim.size_x(), 16);
    assert_eq!(sim.size_y(), 16);
    assert_eq!(sim.steps_per_generation(), 8);
}

#[wasm_bindgen_test]
fn simulator_step_advances_step_counter() {
    let mut sim = Simulator::new(&small_config_json()).unwrap();
    assert_eq!(sim.sim_step(), 0);
    let after = sim.step();
    assert_eq!(after, 1);
    sim.step();
    assert_eq!(sim.sim_step(), 2);
}

#[wasm_bindgen_test]
fn simulator_step_clamps_at_steps_per_generation() {
    let mut sim = Simulator::new(&small_config_json()).unwrap();
    for _ in 0..sim.steps_per_generation() { sim.step(); }
    let saturated = sim.sim_step();
    assert_eq!(saturated, 8);
    assert_eq!(sim.step(), saturated, "step past end is no-op");
}

#[wasm_bindgen_test]
fn simulator_step_generation_runs_remaining_steps() {
    let mut sim = Simulator::new(&small_config_json()).unwrap();
    sim.step();
    let n = sim.step_generation();
    assert_eq!(n, 7);
    assert_eq!(sim.sim_step(), 8);
}

#[wasm_bindgen_test]
fn simulator_get_frame_has_correct_size() {
    let mut sim = Simulator::new(&small_config_json()).unwrap();
    let frame = sim.get_frame().to_vec();
    assert_eq!(frame.len(), 16 * 16 * 4);
}

#[wasm_bindgen_test]
fn simulator_get_frame_alpha_is_always_255() {
    let mut sim = Simulator::new(&small_config_json()).unwrap();
    let frame = sim.get_frame().to_vec();
    for chunk in frame.chunks(4) {
        assert_eq!(chunk[3], 255);
    }
}

#[wasm_bindgen_test]
fn simulator_full_generation_cycle() {
    let mut sim = Simulator::new(&small_config_json()).unwrap();
    sim.step_generation();
    assert_eq!(sim.sim_step(), 8);
    sim.spawn_next_generation().expect("spawn must succeed");
    assert_eq!(sim.generation(), 1);
    assert_eq!(sim.sim_step(), 0);
    assert!(sim.alive_count() > 0);
}

#[wasm_bindgen_test]
fn simulator_run_epoch_combines_steps_and_spawn() {
    let mut sim = Simulator::new(&small_config_json()).unwrap();
    sim.run_epoch().expect("run_epoch must succeed");
    assert_eq!(sim.generation(), 1);
    assert_eq!(sim.sim_step(), 0);
}

#[wasm_bindgen_test]
fn simulator_set_challenge_accepts_valid_json() {
    let mut sim = Simulator::new(&small_config_json()).unwrap();
    let challenge = serde_json::json!({
        "active": ["right_half"], "composition": "Any", "params": {}
    });
    sim.set_challenge(&challenge.to_string()).expect("right_half should be valid");
}

#[wasm_bindgen_test]
fn simulator_set_challenge_rejects_unknown_id() {
    let mut sim = Simulator::new(&small_config_json()).unwrap();
    let challenge = serde_json::json!({
        "active": ["nope"], "composition": "Any", "params": {}
    });
    assert!(sim.set_challenge(&challenge.to_string()).is_err());
}

#[wasm_bindgen_test]
fn simulator_get_stats_returns_object() {
    let sim = Simulator::new(&small_config_json()).unwrap();
    let stats = sim.get_stats().expect("stats must serialize");
    assert!(!stats.is_undefined() && !stats.is_null());
}

#[wasm_bindgen_test]
fn simulator_get_agents_returns_array() {
    let sim = Simulator::new(&small_config_json()).unwrap();
    let agents = sim.get_agents().expect("agents must serialize");
    assert!(js_sys::Array::is_array(&agents));
    let arr = js_sys::Array::from(&agents);
    assert_eq!(arr.length(), 10);
}

#[wasm_bindgen_test]
fn simulator_list_sensors_returns_21_entries() {
    let sim = Simulator::new(&small_config_json()).unwrap();
    let sensors = sim.list_sensors().expect("must serialize");
    let arr = js_sys::Array::from(&sensors);
    assert_eq!(arr.length(), 21, "21 built-in sensors");
}

#[wasm_bindgen_test]
fn simulator_list_actions_returns_17_entries() {
    let sim = Simulator::new(&small_config_json()).unwrap();
    let actions = sim.list_actions().expect("must serialize");
    let arr = js_sys::Array::from(&actions);
    assert_eq!(arr.length(), 17, "17 built-in actions");
}

#[wasm_bindgen_test]
fn simulator_register_js_sensor_increments_count() {
    let mut sim = Simulator::new(&small_config_json()).unwrap();
    let before = js_sys::Array::from(&sim.list_sensors().unwrap()).length();

    let cb = js_sys::Function::new_with_args("agent, simStep", "return 0.5;");
    sim.register_js_sensor("custom_food", "custom food smell", cb);

    let after = js_sys::Array::from(&sim.list_sensors().unwrap()).length();
    assert_eq!(after, before + 1, "registering a JS sensor should grow the registry");
}

#[wasm_bindgen_test]
fn simulator_register_js_action_increments_count() {
    let mut sim = Simulator::new(&small_config_json()).unwrap();
    let before = js_sys::Array::from(&sim.list_actions().unwrap()).length();

    let cb = js_sys::Function::new_with_args("agent, level", "return null;");
    sim.register_js_action("teleport", "teleport home", cb);

    let after = js_sys::Array::from(&sim.list_actions().unwrap()).length();
    assert_eq!(after, before + 1);
}

#[wasm_bindgen_test]
fn simulator_deterministic_with_fixed_seed() {
    let mut sim_a = Simulator::new(&small_config_json()).unwrap();
    let mut sim_b = Simulator::new(&small_config_json()).unwrap();
    sim_a.step_generation();
    sim_b.step_generation();
    assert_eq!(sim_a.get_frame().to_vec(), sim_b.get_frame().to_vec());
}

#[wasm_bindgen_test]
fn simulator_reset_returns_to_generation_0() {
    let mut sim = Simulator::new(&small_config_json()).unwrap();
    sim.run_epoch().unwrap();
    assert_eq!(sim.generation(), 1);
    sim.reset().unwrap();
    assert_eq!(sim.generation(), 0);
    assert_eq!(sim.sim_step(), 0);
}
