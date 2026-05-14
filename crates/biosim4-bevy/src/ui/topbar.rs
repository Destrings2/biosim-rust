//! Top bar: brand, grouped stats, active-challenge chip, telemetry toggle.
//!
//! Stats are arranged into three groups separated by vertical hairlines: SIM
//! (generation / step / alive), PERF (FPS / speed), and SYSTEM (threads /
//! grid). The grid/speed lines that used to live here are now in the canvas
//! frame chrome so they sit closer to what they describe.

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::camera::SimCamera;
use crate::sim::{Sim, SimControls};
use crate::theme;
use crate::ui::{TOPBAR_HEIGHT, UiState};

pub fn draw_topbar(
    mut contexts: EguiContexts,
    sim: Res<Sim>,
    controls: Res<SimControls>,
    cam_q: Query<&Projection, With<SimCamera>>,
    mut ui_state: ResMut<UiState>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    egui::TopBottomPanel::top("topbar")
        .exact_height(TOPBAR_HEIGHT)
        .frame(
            egui::Frame::default()
                .fill(theme::BG)
                .stroke(egui::Stroke::new(1.0, theme::LINE))
                .shadow(theme::dock_shadow())
                .inner_margin(egui::Margin { left: 14, right: 12, top: 0, bottom: 0 }),
        )
        .show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                draw_brand(ui);
                divider(ui);

                // ── Group: SIM — most important stats, accent-tinted when interesting
                let gen = sim.state.generation;
                let step = sim.state.sim_step;
                let total = sim.state.config.steps_per_generation;
                let alive = sim.alive();
                let pop = sim.state.config.population;

                stat(ui, "GEN", theme::mono_value(format!("{gen}")));
                stat(ui, "STEP", theme::mono_value(format!("{step}/{total}")));
                let alive_pct = if pop == 0 { 0.0 } else { (alive as f32) / (pop as f32) };
                let alive_color = if alive_pct > 0.7 {
                    theme::ACCENT
                } else if alive_pct > 0.2 {
                    theme::WARN
                } else {
                    theme::BAD
                };
                stat(
                    ui,
                    "ALIVE",
                    egui::RichText::new(format!("{alive}/{pop}"))
                        .monospace()
                        .size(12.0)
                        .color(alive_color),
                );

                divider(ui);

                // ── Group: PERF
                stat(ui, "FPS", theme::mono_value(format!("{:.0}", controls.fps)));
                stat(ui, "SPEED", theme::mono_value(format!("{}×", controls.speed)));
                stat(ui, "THREADS", theme::mono_value(format!("{}", controls.num_threads)));

                divider(ui);

                // ── Group: VIEW (zoom readout — scroll wheel adjusts it)
                let zoom_px_per_cell = compute_zoom_px_per_cell(controls.pixel_scale, &cam_q);
                stat(ui, "ZOOM", theme::mono_value(format!("{:.1} px/cell", zoom_px_per_cell)));

                // ── Right-aligned cluster
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    telemetry_toggle(ui, &mut ui_state);
                    divider(ui);
                    challenge_chip(ui, &sim, &mut ui_state);
                });
            });
        });
}

fn draw_brand(ui: &mut egui::Ui) {
    ui.scope(|ui| {
        ui.spacing_mut().item_spacing.x = 8.0;

        // Bevel-mark with a soft accent glow underneath.
        let (rect, _resp) = ui.allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::hover());
        let painter = ui.painter();
        theme::accent_glow(painter, rect.center(), 6.0);
        painter.rect_filled(rect, 1.5, theme::ACCENT);
        let inner = rect.shrink(3.0);
        painter.rect_filled(inner, 0.5, theme::BG);

        ui.label(
            egui::RichText::new("BIOSIM4")
                .monospace()
                .size(13.0)
                .strong()
                .color(theme::TEXT),
        );
        ui.label(
            egui::RichText::new("BEVY")
                .size(9.5)
                .color(theme::MUTED)
                .strong(),
        );
    });
}

fn stat(ui: &mut egui::Ui, label: &str, value: egui::RichText) {
    ui.scope(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        ui.add_space(4.0);
        ui.label(theme::key_label(label));
        ui.label(value);
        ui.add_space(4.0);
    });
}

/// Vertical hairline used between stat groups.
fn divider(ui: &mut egui::Ui) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(1.0, 18.0), egui::Sense::hover());
    ui.painter().rect_filled(rect, 0.0, theme::LINE);
}

fn telemetry_toggle(ui: &mut egui::Ui, state: &mut UiState) {
    let on = state.show_telemetry;
    let label = if on { "▾ TELEMETRY" } else { "▸ TELEMETRY" };
    let btn = egui::Button::new(
        egui::RichText::new(label)
            .size(10.5)
            .color(if on { theme::ACCENT } else { theme::TEXT_2 })
            .strong(),
    )
    .fill(egui::Color32::TRANSPARENT)
    .stroke(egui::Stroke::NONE);
    if ui.add(btn).on_hover_text("Toggle the telemetry overlay (last 64 generations)").clicked() {
        state.show_telemetry = !state.show_telemetry;
    }
}

/// Screen pixels covered by one world cell at the camera's current zoom.
/// `pixel_scale` is world-units-per-cell; `ortho.scale` is world-units-per-screen-pixel,
/// so dividing one by the other yields screen-pixels-per-cell.
fn compute_zoom_px_per_cell(
    pixel_scale: f32,
    cam_q: &Query<&Projection, With<SimCamera>>,
) -> f32 {
    let Ok(proj) = cam_q.single() else { return pixel_scale; };
    match proj {
        Projection::Orthographic(o) if o.scale > 1e-6 => pixel_scale / o.scale,
        _ => pixel_scale,
    }
}

fn challenge_chip(ui: &mut egui::Ui, sim: &Sim, state: &mut UiState) {
    let total = sim
        .state
        .challenges
        .schema_list()
        .as_array()
        .map(|a| a.len())
        .unwrap_or(0);
    let active = total > 0;
    let chip = egui::Button::new(
        egui::RichText::new(format!("●  {total} CHALLENGES"))
            .size(10.5)
            .color(if active { theme::ACCENT } else { theme::MUTED })
            .strong(),
    )
    .fill(theme::PANEL_2)
    .stroke(egui::Stroke::new(1.0, theme::LINE))
    .corner_radius(egui::CornerRadius::same(4));
    if ui.add(chip).on_hover_text("Open the challenge picker").clicked() {
        state.show_picker = true;
        state.right_panel_tab = crate::ui::RightPanelTab::Challenge;
    }
}
