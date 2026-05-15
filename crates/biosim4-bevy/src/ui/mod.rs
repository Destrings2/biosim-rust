//! egui-based UI plugin.
//!
//! All panels render inside `Update` after the simulation has stepped and the
//! grid texture has been refreshed, so the UI never reads partial state.

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts, EguiPlugin, EguiPrimaryContextPass, PrimaryEguiContext};

use crate::theme;

pub mod canvas_chrome;
pub mod fast_forward;
pub mod inspector;
pub mod playback;
pub mod right_panel;
pub mod telemetry;
pub mod toolbar;
pub mod topbar;
pub mod widgets;

/// Width of the docked right panel in screen pixels. Camera fit logic uses
/// this to leave room on the right side of the world.
pub const RIGHT_PANEL_WIDTH: f32 = 360.0;
/// Top bar height (matches the value set in `topbar::draw_topbar`).
pub const TOPBAR_HEIGHT: f32 = 44.0;

#[derive(Resource, Default)]
pub struct UiState {
    pub right_panel_tab: RightPanelTab,
    pub theme_installed: bool,
    pub show_telemetry: bool,
    pub show_picker: bool,
    /// Transient banner shown for [`TOAST_VISIBLE_SECS`] after an APPLY
    /// click that didn't reset the sim — surfaces the "this lands next
    /// generation" rule without an interrupting modal.
    pub toast: Option<Toast>,
}

/// Lifespan of an apply-confirmation toast in seconds. Long enough to
/// catch the eye, short enough to disappear before the next interaction.
pub const TOAST_VISIBLE_SECS: f32 = 3.5;

#[derive(Clone)]
pub struct Toast {
    pub message: String,
    pub shown_at: std::time::Instant,
}

impl Toast {
    pub fn new(message: impl Into<String>) -> Self {
        Self { message: message.into(), shown_at: std::time::Instant::now() }
    }

    pub fn is_visible(&self) -> bool {
        self.shown_at.elapsed().as_secs_f32() < TOAST_VISIBLE_SECS
    }
}

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
pub enum RightPanelTab {
    #[default]
    Stats,
    Challenge,
    Breeds,
    Registry,
    Config,
}

impl RightPanelTab {
    /// All tabs in the order they appear on the vertical strip.
    pub const ALL: &'static [RightPanelTab] = &[
        RightPanelTab::Stats,
        RightPanelTab::Challenge,
        RightPanelTab::Breeds,
        RightPanelTab::Registry,
        RightPanelTab::Config,
    ];

    pub fn label(self) -> &'static str {
        match self {
            RightPanelTab::Stats => "STATISTICS",
            RightPanelTab::Challenge => "CHALLENGE",
            RightPanelTab::Breeds => "BREEDS",
            RightPanelTab::Registry => "REGISTRY",
            RightPanelTab::Config => "CONFIG",
        }
    }

    pub fn tooltip(self) -> &'static str {
        match self {
            RightPanelTab::Stats => "Statistics — population, survival, diversity",
            RightPanelTab::Challenge => "Survival challenge picker + parameters",
            RightPanelTab::Breeds => "Breeds — curated sensor/action presets",
            RightPanelTab::Registry => "Enabled sensors and actions",
            RightPanelTab::Config => "Simulation config (population, mutation, world)",
        }
    }

    pub fn icon(self) -> crate::theme::Icon {
        match self {
            RightPanelTab::Stats => crate::theme::Icon::TabStats,
            RightPanelTab::Challenge => crate::theme::Icon::TabChallenge,
            RightPanelTab::Breeds => crate::theme::Icon::TabBreeds,
            RightPanelTab::Registry => crate::theme::Icon::TabRegistry,
            RightPanelTab::Config => crate::theme::Icon::TabConfig,
        }
    }
}

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<EguiPlugin>() {
            app.add_plugins(EguiPlugin::default());
        }
        // Per-system `.run_if(fast_forward_inactive)` skips the heavy
        // egui panels during FF while keeping `telemetry` (user reads it
        // to gauge convergence in real time) and `fast_forward` (progress
        // + cancel button) always on. `install_theme_once` is one-shot
        // and cheap so it stays unconditional.
        let ff_off = crate::sim::fast_forward_inactive;
        app.init_resource::<UiState>()
            .insert_resource(UiState { show_telemetry: true, ..Default::default() })
            // egui systems run in the dedicated egui pass — this is the
            // correct schedule in bevy_egui 0.39+. Running them in plain
            // Update would race the egui pass and lose pointer input.
            .add_systems(
                EguiPrimaryContextPass,
                (
                    install_theme_once,
                    topbar::draw_topbar.run_if(ff_off),
                    right_panel::draw_right_panel.run_if(ff_off),
                    // Frame chrome lives below the floating overlays in z order.
                    canvas_chrome::draw_canvas_chrome.run_if(ff_off),
                    toolbar::draw_floating_toolbar.run_if(ff_off),
                    playback::draw_playback_bar.run_if(ff_off),
                    telemetry::draw_telemetry_overlay,
                    inspector::draw_agent_inspector.run_if(ff_off),
                    // Toast sits above panels but below the FF modal so a
                    // mid-FF kickoff confirmation doesn't get buried.
                    draw_toast.run_if(ff_off),
                    // Modal lives last so it sits above everything else.
                    fast_forward::draw_fast_forward_modal,
                )
                    .chain(),
            );
    }
}

/// Paints the [`UiState::toast`] banner near the bottom of the canvas
/// while it's still within its visible window, then clears it once
/// expired so the next APPLY can replace it without flicker.
fn draw_toast(mut ui_state: ResMut<UiState>, mut contexts: EguiContexts) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    let visible = ui_state.toast.as_ref().is_some_and(|t| t.is_visible());
    if !visible {
        if ui_state.toast.is_some() {
            ui_state.toast = None;
        }
        return;
    }
    let message = ui_state.toast.as_ref().unwrap().message.clone();

    egui::Area::new(egui::Id::new("apply_toast"))
        .anchor(egui::Align2::CENTER_BOTTOM, egui::vec2(-(RIGHT_PANEL_WIDTH * 0.5), -36.0))
        .order(egui::Order::Foreground)
        .interactable(false)
        .show(ctx, |ui| {
            egui::Frame::default()
                .fill(theme::PANEL)
                .stroke(egui::Stroke::new(1.0, theme::ACCENT))
                .corner_radius(egui::CornerRadius::same(6))
                .shadow(theme::float_shadow())
                .inner_margin(egui::Margin::symmetric(14, 10))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        // Accent dot keyed to the row's vertical center.
                        // `allocate_exact_size` gives the icon its own
                        // pixel-precise rect, avoiding the painter-coord
                        // gymnastics that bit me on the first pass.
                        let (rect, _) =
                            ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                        theme::paint_icon(ui.painter(), rect, theme::Icon::Dot, theme::ACCENT);
                        ui.add_space(4.0);
                        ui.label(
                            egui::RichText::new(&message).size(11.5).color(theme::TEXT).strong(),
                        );
                    });
                });
        });

    // Request a repaint while the toast is animating so the fade-out
    // expiry triggers without waiting for the next user input.
    ctx.request_repaint();
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
