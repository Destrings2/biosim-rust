//! egui-based UI plugin.
//!
//! All panels render inside `Update` after the simulation has stepped and the
//! grid texture has been refreshed, so the UI never reads partial state.

use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, PrimaryEguiContext};

use crate::theme;

pub mod canvas_chrome;
pub mod fast_forward;
pub mod inspector;
pub mod playback;
pub mod right_panel;
pub mod telemetry;
pub mod toolbar;
pub mod topbar;

/// Width of the docked right panel in screen pixels. Camera fit logic uses
/// this to leave room on the right side of the world.
pub const RIGHT_PANEL_WIDTH: f32 = 320.0;
/// Top bar height (matches the value set in `topbar::draw_topbar`).
pub const TOPBAR_HEIGHT: f32 = 44.0;

#[derive(Resource, Default)]
pub struct UiState {
    pub right_panel_tab: RightPanelTab,
    pub theme_installed: bool,
    pub show_telemetry: bool,
    pub show_picker: bool,
}

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
pub enum RightPanelTab {
    #[default]
    Stats,
    Challenge,
    Registry,
    Config,
}

impl RightPanelTab {
    pub fn label(self) -> &'static str {
        match self {
            RightPanelTab::Stats => "STATS",
            RightPanelTab::Challenge => "CHALLENGE",
            RightPanelTab::Registry => "REGISTRY",
            RightPanelTab::Config => "CONFIG",
        }
    }
}

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<EguiPlugin>() {
            app.add_plugins(EguiPlugin::default());
        }
        app.init_resource::<UiState>()
            .insert_resource(UiState { show_telemetry: true, ..Default::default() })
            // egui systems run in the dedicated egui pass — this is the
            // correct schedule in bevy_egui 0.39+. Running them in plain
            // Update would race the egui pass and lose pointer input.
            .add_systems(
                EguiPrimaryContextPass,
                (
                    install_theme_once,
                    topbar::draw_topbar,
                    right_panel::draw_right_panel,
                    // Frame chrome lives below the floating overlays in z order.
                    canvas_chrome::draw_canvas_chrome,
                    toolbar::draw_floating_toolbar,
                    playback::draw_playback_bar,
                    telemetry::draw_telemetry_overlay,
                    inspector::draw_agent_inspector,
                    // Modal lives last so it sits above everything else.
                    fast_forward::draw_fast_forward_modal,
                )
                    .chain(),
            );
    }
}

fn install_theme_once(mut ui: ResMut<UiState>, mut contexts: EguiContexts) {
    if ui.theme_installed {
        return;
    }
    if let Ok(ctx) = contexts.ctx_mut() {
        theme::install(ctx);
        ui.theme_installed = true;
    }
}

// Shared widgets live with the panels that use them; the cross-panel helpers
// (`theme::kbd_hint`, `theme::accent_glow`, etc.) live in `crate::theme`.

#[allow(dead_code)]
pub(crate) fn _link_marker(_: PrimaryEguiContext) {}
