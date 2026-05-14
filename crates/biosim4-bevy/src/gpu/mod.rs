//! GPU fast-forward backend.
//!
//! Architecture: GPU-resident state. The sim is uploaded once at FF start;
//! all simulation phases (sensors, neural net, action execution, queue
//! drains, signal fade) run as compute shaders against the same buffers.
//! Only CPU↔GPU sync per generation (for selection + reproduction).
//!
//! See [`fast_forward::GpuFastForward`] for the run loop.
//!
//! ## v1 supported features
//!
//! - Sensors: location (x/y/boundary), last_move_dir, oscillator, age,
//!   random, memory 0–3, barrier_fwd/lr, population (3 variants).
//! - Actions: all move actions, set_responsiveness, set_oscillator_period,
//!   set_longprobe_dist, emit_signal0, write_memory_0..3.
//! - Signal layer 0.
//!
//! ## v1 NOT supported (auto-falls back to CPU)
//!
//! - Signal layer 1 / 2 sensors and emits, signal_*_fwd/lr.
//! - Genetic similarity sensor.
//! - Longprobe (pop/barrier) sensors.
//! - Energy / food.
//! - Active challenges with on_sim_step hooks.
//! - Custom user-registered sensors or actions.
//!
//! When any unsupported feature is enabled, [`GpuFastForward::try_init`]
//! returns `Err` and the FF orchestrator falls back to the CPU path.

use std::sync::{Arc, Mutex};

use bevy::prelude::*;

pub mod context;
pub mod fast_forward;
pub mod pipelines;
pub mod state;

use context::GpuContext;
pub use fast_forward::GpuFastForward;

/// Top-level resource. Either a ready GPU context (FF can attempt GPU
/// init lazily) or a string explaining why the GPU is unavailable.
#[derive(Resource, Clone)]
pub enum GpuAccel {
    Ready { ctx: GpuContext, ff: Arc<Mutex<Option<GpuFastForward>>> },
    Unavailable { reason: String },
}

impl GpuAccel {
    #[allow(dead_code)] // exposed for future system params
    pub fn is_ready(&self) -> bool { matches!(self, GpuAccel::Ready { .. }) }
    #[allow(dead_code)] // exposed for future status UI
    pub fn label(&self) -> String {
        match self {
            GpuAccel::Ready { ctx, .. } => ctx.label(),
            GpuAccel::Unavailable { reason } => reason.clone(),
        }
    }
}

pub struct GpuPlugin;

impl Plugin for GpuPlugin {
    fn build(&self, app: &mut App) {
        let accel = match GpuContext::try_new() {
            Ok(ctx) => {
                info!("biosim4 GPU compute ready ({})", ctx.label());
                GpuAccel::Ready { ctx, ff: Arc::new(Mutex::new(None)) }
            }
            Err(reason) => {
                warn!("biosim4 GPU compute unavailable: {reason}");
                GpuAccel::Unavailable { reason }
            }
        };
        app.insert_resource(accel);
    }
}
