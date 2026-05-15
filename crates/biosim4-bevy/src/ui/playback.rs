//! Floating playback bar — bottom-center of the canvas area.
//!
//! Three logical groups separated by hairlines:
//!   1. Transport: play/pause + step + step-gen + epoch
//!   2. Speed / scale sliders
//!   3. Reset
//!
//! Each transport button shows its keyboard shortcut as a kbd-style hint chip.

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::sim::{SimCommand, SimCommandQueue, SimControls};
use crate::theme;
use crate::ui::RIGHT_PANEL_WIDTH;

pub fn draw_playback_bar(
    mut contexts: EguiContexts,
    mut controls: ResMut<SimControls>,
    mut queue: ResMut<SimCommandQueue>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };

    // Center within the canvas area (left of the right panel).
    let dx = -(RIGHT_PANEL_WIDTH * 0.5);

    egui::Area::new(egui::Id::new("playback_bar"))
        .anchor(egui::Align2::CENTER_BOTTOM, egui::vec2(dx, -18.0))
        .show(ctx, |ui| {
            egui::Frame::default()
                .fill(theme::PANEL)
                .stroke(egui::Stroke::new(1.0, theme::LINE))
                .corner_radius(egui::CornerRadius::same(10))
                .shadow(theme::float_shadow())
                .inner_margin(egui::Margin::symmetric(6, 5))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;

                        // ── Group 1: transport
                        play_button(ui, &mut controls);
                        ghost_button(
                            ui,
                            "STEP",
                            "Space-aware single step",
                            "S",
                            controls.running,
                            || queue.items.push(SimCommand::StepOnce),
                        );
                        ghost_button(
                            ui,
                            "STEP GEN",
                            "Run rest of generation",
                            "G",
                            controls.running,
                            || queue.items.push(SimCommand::StepGeneration),
                        );
                        ghost_button(
                            ui,
                            "EPOCH",
                            "Generation + reproduce",
                            "E",
                            controls.running,
                            || queue.items.push(SimCommand::RunEpoch),
                        );

                        thin_divider(ui);

                        // ── Group 2: speed (zoom is handled by the scroll wheel,
                        // displayed in the top bar)
                        slider_with_label(ui, "SPF", &mut |level| {
                            let mut s = controls.speed as i32;
                            let r = level.add(
                                egui::Slider::new(&mut s, 1..=128)
                                    .logarithmic(true)
                                    .show_value(false),
                            );
                            if r.changed() {
                                controls.speed = s.max(1) as u32;
                            }
                            format!("{}×", controls.speed)
                        });

                        thin_divider(ui);

                        // ── Group 3: reset (danger styling on hover only)
                        let reset_btn = egui::Button::new(
                            egui::RichText::new("RESET")
                                .monospace()
                                .size(10.5)
                                .strong()
                                .color(theme::TEXT_2),
                        )
                        .fill(egui::Color32::TRANSPARENT)
                        .stroke(egui::Stroke::new(1.0, theme::LINE))
                        .corner_radius(egui::CornerRadius::same(5))
                        .min_size(egui::vec2(60.0, 26.0));
                        let r = ui
                            .add(reset_btn)
                            .on_hover_text("Restart at generation 0 with current config");
                        if r.hovered() {
                            // Repaint with red highlight by drawing a stroke overlay.
                            ui.painter().rect_stroke(
                                r.rect.expand(0.0),
                                egui::CornerRadius::same(5),
                                egui::Stroke::new(1.0, theme::BAD),
                                egui::StrokeKind::Inside,
                            );
                        }
                        if r.clicked() {
                            queue.items.push(SimCommand::Reset);
                        }
                    });
                });
        });
}

fn play_button(ui: &mut egui::Ui, controls: &mut SimControls) {
    let running = controls.running;
    let label = if running { "⏸  PAUSE" } else { "▶  PLAY" };
    let color = if running { theme::WARN } else { theme::ACCENT };
    let stroke = if running { theme::WARN } else { theme::ACCENT };
    let btn =
        egui::Button::new(egui::RichText::new(label).monospace().size(11.0).strong().color(color))
            .fill(egui::Color32::from_rgba_premultiplied(0, 0, 0, 0))
            .stroke(egui::Stroke::new(1.0, stroke))
            .corner_radius(egui::CornerRadius::same(5))
            .min_size(egui::vec2(88.0, 26.0));
    let r = ui.add(btn);
    if r.clicked() {
        controls.running = !controls.running;
    }
    ui.add_space(2.0);
    theme::kbd_hint(ui, "Space");
}

fn ghost_button(
    ui: &mut egui::Ui,
    label: &str,
    tip: &str,
    kbd: &str,
    disabled: bool,
    mut on_click: impl FnMut(),
) {
    let btn = egui::Button::new(
        egui::RichText::new(label)
            .monospace()
            .size(10.5)
            .color(if disabled { theme::MUTED } else { theme::TEXT_2 })
            .strong(),
    )
    .fill(egui::Color32::TRANSPARENT)
    .stroke(egui::Stroke::new(1.0, theme::LINE))
    .corner_radius(egui::CornerRadius::same(5))
    .min_size(egui::vec2(64.0, 26.0));
    let r = ui.add_enabled(!disabled, btn).on_hover_text(tip);
    if r.clicked() {
        on_click();
    }
    ui.add_space(2.0);
    theme::kbd_hint(ui, kbd);
}

fn slider_with_label(
    ui: &mut egui::Ui,
    label: &str,
    body: &mut dyn FnMut(&mut egui::Ui) -> String,
) {
    ui.scope(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        ui.label(theme::key_label(label));
        let value = body(ui);
        ui.label(egui::RichText::new(value).monospace().size(11.0).color(theme::TEXT));
    });
}

fn thin_divider(ui: &mut egui::Ui) {
    ui.add_space(4.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(1.0, 22.0), egui::Sense::hover());
    ui.painter().rect_filled(rect, 0.0, theme::LINE);
    ui.add_space(4.0);
}
