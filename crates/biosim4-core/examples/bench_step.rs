//! Headless throughput bench.
//!
//! Usage: `cargo run -p biosim4-core --example bench_step --release [threads]`
//! Runs a fixed configuration for a fixed number of generations and prints
//! wall time + steps/sec. Default 4 threads; pass `1` for the deterministic
//! single-thread path.

use std::time::Instant;

use biosim4_core::{
    sim_config::SimConfig,
    sim_state::SimulationState,
    sim_step::step_generation,
    spawn::{initialize_generation_0, spawn_new_generation},
};

fn main() {
    let threads: u32 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(4);
    let population: u32 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(1500);

    let gens: u32 = 30;
    let mut cfg = SimConfig::default();
    cfg.rng_seed = 0xB105F00D;
    cfg.num_threads = threads;
    cfg.population = population;
    cfg.size_x = 128;
    cfg.size_y = 128;
    cfg.steps_per_generation = 200;
    cfg.max_generations = gens;
    cfg.signal_layers = 1;
    cfg.barrier_type = 0;
    cfg.enable_energy = false;

    let mut state = SimulationState::new(cfg.clone());
    biosim4_sensors::register_builtin_sensors(&mut state.sensors);
    biosim4_actions::register_builtin_actions(&mut state.actions);
    initialize_generation_0(&mut state);

    let mut step_time = std::time::Duration::ZERO;
    let mut spawn_time = std::time::Duration::ZERO;
    let mut per_gen_step_ms: Vec<f32> = Vec::with_capacity(gens as usize);
    let start = Instant::now();
    for _ in 0..gens {
        let t = Instant::now();
        step_generation(&mut state);
        let dt = t.elapsed();
        step_time += dt;
        per_gen_step_ms.push(dt.as_secs_f32() * 1000.0);
        let t = Instant::now();
        let _ = spawn_new_generation(&mut state);
        spawn_time += t.elapsed();
    }
    let elapsed = start.elapsed();
    let first_5 = per_gen_step_ms.iter().take(5).sum::<f32>() / 5.0;
    let last_5 = per_gen_step_ms.iter().rev().take(5).sum::<f32>() / 5.0;

    let total_steps = (gens as u64) * (cfg.steps_per_generation as u64);
    let steps_per_sec = total_steps as f64 / elapsed.as_secs_f64();
    println!(
        "threads={threads}  pop={pop}  grid={x}x{y}  gens={gens}  elapsed={elapsed:?}  step={st:?}  spawn={sp:?}  steps/sec={sps:.0}  step_first5={f5:.1}ms  step_last5={l5:.1}ms",
        pop = cfg.population,
        x = cfg.size_x,
        y = cfg.size_y,
        st = step_time,
        sp = spawn_time,
        sps = steps_per_sec,
        f5 = first_5,
        l5 = last_5,
    );
}
