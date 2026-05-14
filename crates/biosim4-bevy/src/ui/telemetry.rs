//! Telemetry overlay — compact single-row strip with three inline metrics,
//! anchored above the playback bar.
//!
//! Each metric reads: LABEL  VALUE  [sparkline]. Total height is ~40px so the
//! overlay doesn't crowd the canvas.

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::sim::SimHistory;
use crate::theme;
use crate::ui::{UiState, RIGHT_PANEL_WIDTH};

pub fn draw_telemetry_overlay(
    mut contexts: EguiContexts,
    history: Res<SimHistory>,
    ui_state: Res<UiState>,
) {
    if !ui_state.show_telemetry { return; }
    let Ok(ctx) = contexts.ctx_mut() else { return };

    let dx = -(RIGHT_PANEL_WIDTH * 0.5);

    egui::Area::new(egui::Id::new("telemetry"))
        .anchor(egui::Align2::CENTER_BOTTOM, egui::vec2(dx, -68.0))
        .show(ctx, |ui| {
            egui::Frame::default()
                .fill(theme::PANEL)
                .stroke(egui::Stroke::new(1.0, theme::LINE))
                .corner_radius(egui::CornerRadius::same(8))
                .shadow(theme::float_shadow())
                .inner_margin(egui::Margin::symmetric(12, 6))
                .show(ui, |ui| {
                    if history.points.is_empty() {
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new("TELEMETRY")
                                    .size(9.5)
                                    .color(theme::MUTED)
                                    .strong(),
                            );
                            ui.label(
                                egui::RichText::new("·  waiting for first epoch")
                                    .size(10.5)
                                    .color(theme::TEXT_2)
                                    .italics(),
                            );
                        });
                        return;
                    }

                    let latest = history.latest().cloned().unwrap_or_default();
                    let survival:  Vec<f32> = history.points.iter().map(|p| p.survival_rate).collect();
                    let diversity: Vec<f32> = history.points.iter().map(|p| p.diversity).collect();
                    let alive:     Vec<f32> = history.points.iter().map(|p| p.alive as f32).collect();
                    let max_alive = alive.iter().cloned().fold(0.0_f32, f32::max).max(1.0);

                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 14.0;
                        ui.label(
                            egui::RichText::new(format!("TELEMETRY  ·  GEN {}", latest.generation))
                                .size(9.5)
                                .color(theme::MUTED)
                                .strong(),
                        );
                        inline_metric(
                            ui,
                            "SURVIVAL",
                            format!("{:>5.1}%", latest.survival_rate * 100.0),
                            &survival,
                            theme::ACCENT,
                            0.0..1.0,
                        );
                        inline_metric(
                            ui,
                            "DIVERSITY",
                            format!("{:>5.3}", latest.diversity),
                            &diversity,
                            theme::WARN,
                            0.0..1.0,
                        );
                        inline_metric(
                            ui,
                            "POP",
                            format!("{:>5}", latest.alive),
                            &alive,
                            theme::TEXT_2,
                            0.0..max_alive,
                        );
                    });
                });
        });
}

fn inline_metric(
    ui: &mut egui::Ui,
    label: &str,
    value: String,
    pts: &[f32],
    color: egui::Color32,
    range: std::ops::Range<f32>,
) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 5.0;
        ui.label(
            egui::RichText::new(label)
                .size(9.5)
                .color(theme::MUTED)
                .strong(),
        );
        ui.label(
            egui::RichText::new(value)
                .monospace()
                .size(11.5)
                .color(theme::TEXT)
                .strong(),
        );

        let (rect, _) = ui.allocate_exact_size(egui::vec2(86.0, 18.0), egui::Sense::hover());
        let painter = ui.painter();
        painter.rect_filled(rect, 3.0, theme::BG_2);
        if pts.len() > 1 {
            let span = (range.end - range.start).max(1e-6);
            let xs: Vec<egui::Pos2> = pts
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let nx = i as f32 / (pts.len() - 1) as f32;
                    let ny = ((v - range.start) / span).clamp(0.0, 1.0);
                    egui::pos2(
                        rect.left() + nx * rect.width(),
                        rect.bottom() - 1.0 - ny * (rect.height() - 3.0),
                    )
                })
                .collect();

            let mut fill_pts = xs.clone();
            fill_pts.push(egui::pos2(rect.right(), rect.bottom()));
            fill_pts.push(egui::pos2(rect.left(), rect.bottom()));
            painter.add(egui::Shape::convex_polygon(
                fill_pts,
                color.gamma_multiply(0.18),
                egui::Stroke::NONE,
            ));
            painter.add(egui::Shape::line(xs.clone(), egui::Stroke::new(1.2, color)));
            if let Some(&last) = xs.last() {
                painter.circle_filled(last, 2.0, color);
            }
        }
    });
}
