//! Compute pipelines for the GPU step shader.
//!
//! One bind-group layout binds every storage buffer + the params uniform;
//! the same bind group is reused across every dispatch in a step. Pipelines
//! differ only in which entry point of `step.wgsl` they call.

use super::context::GpuContext;
use super::state::GpuState;

pub struct Pipelines {
    pub bind_group: wgpu::BindGroup,
    pub clear_step_scratch: wgpu::ComputePipeline,
    pub phase1_sensors_ff: wgpu::ComputePipeline,
    pub phase2_actions: wgpu::ComputePipeline,
    pub drain_deaths: wgpu::ComputePipeline,
    pub drain_moves: wgpu::ComputePipeline,
    pub signal_fade: wgpu::ComputePipeline,
}

impl Pipelines {
    pub fn new(ctx: &GpuContext, state: &GpuState) -> Self {
        let device = &ctx.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("biosim4 step.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/step.wgsl").into(),
            ),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("biosim4 step bgl"),
            entries: &[
                storage_rw(0),  // agents
                storage_ro(1),  // agent_nets
                storage_ro(2),  // connections
                storage_rw(3),  // grid (atomic)
                storage_rw(4),  // signals (atomic)
                storage_rw(5),  // action_levels
                storage_rw(6),  // move_queue_count (atomic)
                storage_rw(7),  // move_queue
                storage_rw(8),  // death_queue_count (atomic)
                storage_rw(9),  // death_queue
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_ro(11), // food
                storage_ro(12), // genome_data
                storage_ro(13), // genome_offsets
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("biosim4 step layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let mk = |entry: &'static str, label: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("biosim4 step bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0,  resource: state.buf_agents.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1,  resource: state.buf_agent_nets.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2,  resource: state.buf_connections.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3,  resource: state.buf_grid.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4,  resource: state.buf_signals.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5,  resource: state.buf_action_levels.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6,  resource: state.buf_move_queue_count.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7,  resource: state.buf_move_queue.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8,  resource: state.buf_death_queue_count.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9,  resource: state.buf_death_queue.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: state.buf_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: state.buf_food.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: state.buf_genome_data.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 13, resource: state.buf_genome_offsets.as_entire_binding() },
            ],
        });

        Self {
            bind_group,
            clear_step_scratch: mk("clear_step_scratch", "clear_step_scratch"),
            phase1_sensors_ff: mk("phase1_sensors_ff",   "phase1_sensors_ff"),
            phase2_actions:    mk("phase2_actions",      "phase2_actions"),
            drain_deaths:      mk("drain_deaths",        "drain_deaths"),
            drain_moves:       mk("drain_moves",         "drain_moves"),
            signal_fade:       mk("signal_fade",         "signal_fade"),
        }
    }
}

fn storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
