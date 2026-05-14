//! biosim4-bevy — native Bevy frontend for biosim4-rs.
//!
//! The simulation runs on the main thread; parallelism comes from rayon
//! inside `biosim4-core` (enabled here via the `parallel` feature). Bevy's
//! own task pool is reserved for rendering and asset work, which keeps the
//! per-frame cost predictable.
//!
//! Module layout:
//!   - `sim`           — owns the [`SimulationState`] resource and stepping.
//!   - `grid_render`   — uploads the grid to a single sprite-backed texture.
//!   - `camera`        — 2D camera with pan + zoom.
//!   - `tool`          — translates mouse clicks into [`SimCommand`]s.
//!   - `ui`            — egui panels (top bar, side panel, playback, inspector).
//!   - `theme`         — palette + style installation.

mod camera;
mod grid_render;
mod sim;
mod theme;
mod tool;
mod ui;

use bevy::prelude::*;
use bevy::window::{PresentMode, Window, WindowTheme};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "biosim4 · bevy".into(),
                resolution: (1480u32, 900u32).into(),
                resizable: true,
                present_mode: PresentMode::AutoVsync,
                window_theme: Some(WindowTheme::Dark),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::srgb(0.04, 0.045, 0.055)))
        .add_plugins((
            sim::SimPlugin,
            grid_render::GridRenderPlugin,
            camera::CameraPlugin,
            tool::ToolPlugin,
            ui::UiPlugin,
        ))
        .run();
}
