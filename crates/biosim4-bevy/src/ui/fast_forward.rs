//! Fast-forward progress modal — shown while [`FastForward`] is active.
//!
//! Renders a centered window with a progress bar (gen X / Y), an elapsed/ETA
//! readout, and a cancel button. The bevy `step_simulation` system is doing
//! the actual work in tight time-bounded slices, so this modal updates ~60×
//! per second giving the user a live progress indicator.

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::sim::{FastForward, SimCommand, SimCommandQueue};
use crate::theme;

pub fn draw_fast_forward_modal(
    mut contexts: EguiContexts,
    fast_forward: Res<FastForward>,
    mut queue: ResMut<SimCommandQueue>,
) {
    let Some(state) = fast_forward.active.as_ref() else { return };
    let Ok(ctx) = contexts.ctx_mut() else { return };

    // A soft dim overlay behind the modal — egui doesn't have a built-in
    // backdrop, so we paint a semi-opaque rect over the full viewport.
    let screen = ctx.viewport_rect();
    egui::Area::new(egui::Id::new("ff_backdrop"))
        .order(egui::Order::Foreground)
        .interactable(false)
        .fixed_pos(screen.min)
        .show(ctx, |ui| {
            ui.painter().rect_filled(screen, 0.0, egui::Color32::from_black_alpha(140));
        });

    egui::Area::new(egui::Id::new("ff_modal"))
        .order(egui::Order::Tooltip)
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ctx, |ui| {
            egui::Frame::default()
                .fill(theme::PANEL)
                .stroke(egui::Stroke::new(1.0, theme::ACCENT))
                .corner_radius(egui::CornerRadius::same(10))
                .shadow(theme::float_shadow())
                .inner_margin(egui::Margin::same(20))
                .show(ui, |ui| {
                    ui.set_min_width(420.0);

                    ui.label(
                        egui::RichText::new("FAST FORWARD")
                            .size(10.0)
                            .color(theme::ACCENT)
                            .strong(),
                    );
                    ui.label(
                        egui::RichText::new(format!(
                            "Generation {} → {}",
                            state.start_gen, state.target_gen
                        ))
                        .monospace()
                        .size(13.0)
                        .color(theme::TEXT),
                    );
                    ui.add_space(8.0);

                    // ── Progress bar
                    let progress = state.progress();
                    let (rect, _) = ui.allocate_exact_size(
                        egui::vec2(ui.available_width(), 10.0),
                        egui::Sense::hover(),
                    );
                    let painter = ui.painter();
                    painter.rect_filled(rect, 5.0, theme::BG_2);
                    let fill_w = rect.width() * progress;
                    if fill_w > 1.0 {
                        let fill_rect =
                            egui::Rect::from_min_size(rect.min, egui::vec2(fill_w, rect.height()));
                        painter.rect_filled(fill_rect, 5.0, theme::ACCENT);
                    }
                    painter.rect_stroke(
                        rect,
                        5.0,
                        egui::Stroke::new(1.0, theme::LINE),
                        egui::StrokeKind::Outside,
                    );

                    ui.add_space(8.0);

                    // ── Readouts row
                    ui.horizontal(|ui| {
                        readout(
                            ui,
                            "DONE",
                            format!(
                                "{} / {}  ({:.0}%)",
                                state.done_count(),
                                state.total(),
                                progress * 100.0
                            ),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let elapsed = format_duration(state.elapsed());
                            let eta =
                                state.eta().map(format_duration).unwrap_or_else(|| "—".into());
                            readout(ui, "ETA", eta);
                            ui.add_space(16.0);
                            readout(ui, "ELAPSED", elapsed);
                        });
                    });

                    let done = state.done_count();
                    if done > 0 {
                        let per_gen_ms = state.elapsed().as_secs_f64() * 1000.0 / done as f64;
                        let gen_per_s = if per_gen_ms > 0.0 { 1000.0 / per_gen_ms } else { 0.0 };
                        ui.add_space(2.0);
                        ui.label(
                            egui::RichText::new(format!(
                                "{:.1} ms/gen  ·  {:.1} gens/sec  ·  rendering paused",
                                per_gen_ms, gen_per_s,
                            ))
                            .size(10.5)
                            .color(theme::MUTED)
                            .italics(),
                        );
                    } else {
                        ui.add_space(2.0);
                        ui.label(
                            egui::RichText::new("Spinning up — rendering paused.")
                                .size(10.5)
                                .color(theme::MUTED)
                                .italics(),
                        );
                    }

                    ui.add_space(14.0);

                    // ── Cancel
                    let btn = egui::Button::new(
                        egui::RichText::new("CANCEL").size(11.0).strong().color(theme::TEXT),
                    )
                    .fill(theme::PANEL_2)
                    .stroke(egui::Stroke::new(1.0, theme::BAD))
                    .corner_radius(egui::CornerRadius::same(5))
                    .min_size(egui::vec2(100.0, 28.0));
                    if ui.add(btn).clicked() {
                        queue.items.push(SimCommand::CancelFastForward);
                    }
                });
        });
}

fn readout(ui: &mut egui::Ui, label: &str, value: String) {
    ui.scope(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        ui.label(egui::RichText::new(label).size(9.5).color(theme::MUTED).strong());
        ui.label(egui::RichText::new(value).monospace().size(11.5).color(theme::TEXT));
    });
}

fn format_duration(d: std::time::Duration) -> String {
    let total = d.as_secs_f64();
    if total < 1.0 {
        format!("{}ms", (total * 1000.0).round() as i64)
    } else if total < 60.0 {
        format!("{:.1}s", total)
    } else if total < 3600.0 {
        let m = (total / 60.0).floor() as i64;
        let s = (total - (m as f64 * 60.0)).round() as i64;
        format!("{m}m{s:02}s")
    } else {
        let h = (total / 3600.0).floor() as i64;
        let m = ((total - h as f64 * 3600.0) / 60.0).floor() as i64;
        format!("{h}h{m:02}m")
    }
}
