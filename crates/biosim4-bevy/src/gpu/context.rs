//! Standalone wgpu compute device for the GPU fast-forward backend.
//!
//! Not shared with Bevy's render device — Bevy's lives inside its render
//! world behind an `Arc<RenderDevice>` that isn't trivially reachable from a
//! main-world system, and the FF compute work touches none of Bevy's render
//! resources. Spinning up our own device keeps lifetime management simple
//! and the GPU pipeline self-contained.

use std::sync::Arc;

#[derive(Clone)]
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub adapter_info: wgpu::AdapterInfo,
}

impl GpuContext {
    pub fn try_new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ))
        .map_err(|e| format!("no compatible GPU adapter: {e}"))?;

        let adapter_limits = adapter.limits();
        let info = adapter.get_info();

        // The compute pipelines need 13 storage buffers per stage (agents,
        // grid, signals, connections, food, genome_data, genome_offsets, ...).
        // The downlevel default is 4 so we have to bump that or
        // request_device will reject the layout.
        let mut required_limits = wgpu::Limits::downlevel_defaults();
        required_limits.max_storage_buffers_per_shader_stage = adapter_limits
            .max_storage_buffers_per_shader_stage
            .max(13);
        // Storage buffer binding size — our agent struct is large and total
        // buffers can run to many MB. Cap at adapter's max.
        required_limits.max_storage_buffer_binding_size = adapter_limits
            .max_storage_buffer_binding_size;
        required_limits.max_buffer_size = adapter_limits.max_buffer_size;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("biosim4-gpu device"),
                required_features: wgpu::Features::empty(),
                required_limits,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            },
        ))
        .map_err(|e| format!("request_device failed: {e}"))?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info: info,
        })
    }

    pub fn label(&self) -> String {
        let backend = match self.adapter_info.backend {
            wgpu::Backend::Metal => "Metal",
            wgpu::Backend::Vulkan => "Vulkan",
            wgpu::Backend::Dx12 => "DX12",
            wgpu::Backend::Gl => "OpenGL",
            wgpu::Backend::BrowserWebGpu => "WebGPU",
            _ => "Unknown",
        };
        format!("{} · {backend}", self.adapter_info.name)
    }
}
