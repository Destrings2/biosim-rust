//! GPU fast-forward orchestrator.
//!
//! Per FF run:
//!
//! ```text
//! 1. upload SimulationState to GPU
//! 2. for each remaining generation:
//!      for each remaining step:
//!        queue: clear_scratch → phase1 → phase2 → drain_deaths
//!               → drain_moves → signal_fade
//!      submit + poll
//!      download agents
//!      CPU: spawn_new_generation(state)
//!      upload new generation
//! ```
//!
//! Within a generation we batch every step's dispatches into a single
//! command encoder, so the GPU just chews through them without any CPU
//! intervention. The CPU only blocks (poll) once per generation when it
//! needs the agents back for selection + reproduction.

use std::time::Instant;

use biosim4_core::{
    analysis::collect_epoch_stats,
    sim_state::SimulationState,
    spawn::spawn_new_generation,
};

use super::context::GpuContext;
use super::pipelines::Pipelines;
use super::state::{GpuState, IdMaps, ParamsGpu, WORKGROUP};

pub struct GenerationOutcome {
    pub generation_finished: u32,
    pub survivors: u32,
    pub survival_rate: f32,
    pub diversity: f32,
}

pub struct GpuFastForward {
    ctx: GpuContext,
    pipelines: Pipelines,
    state: GpuState,
    maps: IdMaps,
}

impl GpuFastForward {
    pub fn try_init(ctx: GpuContext, state: &SimulationState) -> Result<Self, String> {
        let maps = IdMaps::build(state);
        if !maps.all_supported(state) {
            return Err(
                "the current registry config uses sensors or actions not yet \
                 ported to GPU; falling back to CPU".to_string(),
            );
        }
        let gpu_state = GpuState::new(ctx.clone(), state, &maps);
        let pipelines = Pipelines::new(&ctx, &gpu_state);
        Ok(Self { ctx, pipelines, state: gpu_state, maps })
    }

    /// Upload, run one generation's worth of steps on GPU, download.
    /// Caller runs `spawn_new_generation` on CPU between calls and then
    /// invokes `upload_after_rollover` so we restart the next gen with
    /// fresh nets.
    pub fn run_one_generation(&mut self, state: &mut SimulationState) -> GenerationOutcome {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let total = state.config.steps_per_generation;

        // We always start a generation at sim_step = 0. (Mid-generation FF
        // resume isn't supported on the GPU path — we'd need to start
        // from the current sim_step, but the CPU FF orchestrator only
        // calls us at gen boundaries.)
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu generation"),
        });

        // For each step, queue the 6 dispatches.
        for step in 0..total {
            // Refresh params for this step. write_buffer doesn't go on the
            // encoder; it queues into the device. Order-relative to compute
            // submissions is FIFO on the queue.
            let params = ParamsGpu {
                num_population: self.state.population,
                sim_step: step,
                generation: state.generation,
                size_x: self.state.size_x,
                size_y: self.state.size_y,
                steps_per_generation: total,
                _sensor_count: 0,
                action_count: self.state.action_count,
                pop_radius: state.config.population_sensor_radius,
                rng_seed_lo: state.config.rng_seed as u32,
                rng_seed_hi: (state.config.rng_seed >> 32) as u32,
                short_probe_distance: state.config.short_probe_barrier_distance,
                signal_layers: self.state.signal_layers,
                food_regen_rate: state.config.food_regen_rate,
                energy_per_step_cost: state.config.energy_per_step_cost,
                _pad: 0,
            };
            queue.write_buffer(
                &self.state.buf_params,
                0,
                bytemuck::bytes_of(&params),
            );

            // Each dispatch is its own compute pass.
            {
                let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("clear_scratch"),
                    timestamp_writes: None,
                });
                cp.set_pipeline(&self.pipelines.clear_step_scratch);
                cp.set_bind_group(0, &self.pipelines.bind_group, &[]);
                let total = self.state.population * self.state.action_count;
                cp.dispatch_workgroups(total.div_ceil(WORKGROUP).max(1), 1, 1);
            }
            {
                let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("phase1"), timestamp_writes: None,
                });
                cp.set_pipeline(&self.pipelines.phase1_sensors_ff);
                cp.set_bind_group(0, &self.pipelines.bind_group, &[]);
                cp.dispatch_workgroups(self.state.population.div_ceil(WORKGROUP).max(1), 1, 1);
            }
            {
                let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("phase2"), timestamp_writes: None,
                });
                cp.set_pipeline(&self.pipelines.phase2_actions);
                cp.set_bind_group(0, &self.pipelines.bind_group, &[]);
                cp.dispatch_workgroups(self.state.population.div_ceil(WORKGROUP).max(1), 1, 1);
            }
            {
                let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("drain_deaths"), timestamp_writes: None,
                });
                cp.set_pipeline(&self.pipelines.drain_deaths);
                cp.set_bind_group(0, &self.pipelines.bind_group, &[]);
                cp.dispatch_workgroups(self.state.population.div_ceil(WORKGROUP).max(1), 1, 1);
            }
            {
                let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("drain_moves"), timestamp_writes: None,
                });
                cp.set_pipeline(&self.pipelines.drain_moves);
                cp.set_bind_group(0, &self.pipelines.bind_group, &[]);
                cp.dispatch_workgroups(self.state.population.div_ceil(WORKGROUP).max(1), 1, 1);
            }
            {
                let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("signal_fade"), timestamp_writes: None,
                });
                cp.set_pipeline(&self.pipelines.signal_fade);
                cp.set_bind_group(0, &self.pipelines.bind_group, &[]);
                let cells = self.state.size_x * self.state.size_y * self.state.signal_layers.max(1);
                cp.dispatch_workgroups(cells.div_ceil(WORKGROUP).max(1), 1, 1);
            }
        }

        queue.submit(Some(encoder.finish()));
        // Poll once after all steps have been queued.
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        // Pull state back to CPU.
        self.state.download_agents(state);
        state.sim_step = total;

        // Selection + reproduction on CPU.
        let prev_gen = state.generation;
        let survivors = spawn_new_generation(state);
        let stats = collect_epoch_stats(state, survivors);

        // Re-upload the brand-new generation's nets/positions.
        self.state.upload_all(state, &self.maps);

        GenerationOutcome {
            generation_finished: prev_gen,
            survivors,
            survival_rate: stats.survival_rate(),
            diversity: stats.diversity,
        }
    }

    /// Initial upload — call once at the start of a fast-forward run. Idempotent.
    pub fn initial_upload(&mut self, state: &SimulationState) {
        self.state.upload_all(state, &self.maps);
    }

    /// Convenience: run `n` consecutive generations to completion, returning
    /// per-generation outcomes. Used for the synchronous fast-forward modal.
    #[allow(dead_code)] // exposed for future headless callers
    pub fn run_n_generations(
        &mut self,
        state: &mut SimulationState,
        n: u32,
        mut on_generation: impl FnMut(&GenerationOutcome, std::time::Duration),
    ) {
        self.initial_upload(state);
        for _ in 0..n {
            let t = Instant::now();
            let outcome = self.run_one_generation(state);
            on_generation(&outcome, t.elapsed());
        }
    }
}
