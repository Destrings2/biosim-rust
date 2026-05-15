//! Visual design tokens shared across the egui UI.
//!
//! Mirrors the dark + green-accent palette from the web frontend
//! (`frontend/src/styles.css`) so the two clients feel like the same product.

use bevy_egui::egui;

pub const BG: egui::Color32 = egui::Color32::from_rgb(0x0a, 0x0b, 0x0d);
pub const BG_2: egui::Color32 = egui::Color32::from_rgb(0x0f, 0x11, 0x14);
pub const PANEL: egui::Color32 = egui::Color32::from_rgb(0x14, 0x16, 0x1a);
pub const PANEL_2: egui::Color32 = egui::Color32::from_rgb(0x1a, 0x1d, 0x22);
pub const LINE: egui::Color32 = egui::Color32::from_rgb(0x23, 0x27, 0x2e);
pub const LINE_2: egui::Color32 = egui::Color32::from_rgb(0x2d, 0x32, 0x3b);
pub const TEXT: egui::Color32 = egui::Color32::from_rgb(0xe6, 0xe8, 0xec);
pub const TEXT_2: egui::Color32 = egui::Color32::from_rgb(0xa8, 0xae, 0xb9);
pub const MUTED: egui::Color32 = egui::Color32::from_rgb(0x6b, 0x71, 0x7c);
pub const ACCENT: egui::Color32 = egui::Color32::from_rgb(0x7d, 0xd3, 0xa8);
pub const ACCENT_SOFT: egui::Color32 =
    egui::Color32::from_rgba_premultiplied(0x32, 0x55, 0x44, 0x70);
pub const WARN: egui::Color32 = egui::Color32::from_rgb(0xe8, 0xa8, 0x7c);
pub const BAD: egui::Color32 = egui::Color32::from_rgb(0xe0, 0x7b, 0x7b);

/// Apply the biosim theme to an egui context. Called once at startup.
pub fn install(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.override_text_color = Some(TEXT);
    visuals.window_fill = PANEL;
    visuals.window_stroke = egui::Stroke::new(1.0, LINE);
    visuals.window_corner_radius = egui::CornerRadius::same(6);
    visuals.panel_fill = BG;
    visuals.faint_bg_color = PANEL_2;
    visuals.extreme_bg_color = BG_2;
    visuals.code_bg_color = BG_2;
    visuals.selection.bg_fill = ACCENT_SOFT;
    visuals.selection.stroke = egui::Stroke::new(1.0, ACCENT);
    visuals.hyperlink_color = ACCENT;
    visuals.warn_fg_color = WARN;
    visuals.error_fg_color = BAD;

    // Widget visuals — base inactive state.
    let widget_corner = egui::CornerRadius::same(4);
    visuals.widgets.noninteractive.bg_fill = PANEL;
    visuals.widgets.noninteractive.weak_bg_fill = PANEL;
    visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, LINE);
    visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, TEXT_2);
    visuals.widgets.noninteractive.corner_radius = widget_corner;

    visuals.widgets.inactive.bg_fill = PANEL_2;
    visuals.widgets.inactive.weak_bg_fill = PANEL_2;
    visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, LINE);
    visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, TEXT_2);
    visuals.widgets.inactive.corner_radius = widget_corner;

    visuals.widgets.hovered.bg_fill = LINE;
    visuals.widgets.hovered.weak_bg_fill = LINE;
    visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, LINE_2);
    visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, TEXT);
    visuals.widgets.hovered.corner_radius = widget_corner;

    visuals.widgets.active.bg_fill = ACCENT_SOFT;
    visuals.widgets.active.weak_bg_fill = ACCENT_SOFT;
    visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, ACCENT);
    visuals.widgets.active.fg_stroke = egui::Stroke::new(1.0, ACCENT);
    visuals.widgets.active.corner_radius = widget_corner;

    visuals.widgets.open.bg_fill = PANEL_2;
    visuals.widgets.open.weak_bg_fill = PANEL_2;
    visuals.widgets.open.bg_stroke = egui::Stroke::new(1.0, ACCENT);
    visuals.widgets.open.fg_stroke = egui::Stroke::new(1.0, TEXT);
    visuals.widgets.open.corner_radius = widget_corner;

    ctx.set_visuals(visuals);

    // Tighten the default spacing to feel like a dense pro tool.
    let mut style: egui::Style = (*ctx.style()).clone();
    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    style.spacing.button_padding = egui::vec2(8.0, 4.0);
    style.spacing.window_margin = egui::Margin::symmetric(12, 10);
    style.spacing.indent = 14.0;
    style.spacing.slider_width = 140.0;
    ctx.set_style(style);
}

/// Convert an RGB color (0..=255) to an egui Color32. Helper for agent colors.
#[inline]
pub fn rgb(c: [u8; 3]) -> egui::Color32 {
    egui::Color32::from_rgb(c[0], c[1], c[2])
}

/// Standard panel shadow (subtle elevation for floating overlays).
pub fn float_shadow() -> egui::Shadow {
    egui::Shadow {
        offset: [0, 8],
        blur: 28,
        spread: 0,
        color: egui::Color32::from_black_alpha(140),
    }
}

/// Tight inset shadow used by docked panels (top bar, side panel).
pub fn dock_shadow() -> egui::Shadow {
    egui::Shadow { offset: [0, 2], blur: 6, spread: 0, color: egui::Color32::from_black_alpha(60) }
}

/// "10px UPPERCASE  ACCENT-TINT  STRONG" key label used on stat chips.
#[inline]
pub fn key_label(text: &str) -> egui::RichText {
    egui::RichText::new(text).size(10.0).color(MUTED).strong()
}

/// Monospace tabular value (uses egui's default tnum-friendly mono font).
#[inline]
pub fn mono_value(text: impl Into<String>) -> egui::RichText {
    egui::RichText::new(text.into()).monospace().size(12.0).color(TEXT)
}

/// Render a small "kbd"-style hint chip next to a button label.
pub fn kbd_hint(ui: &mut egui::Ui, text: &str) {
    let visuals = ui.visuals();
    let _ = visuals; // future styling hook
    egui::Frame::default()
        .fill(BG_2)
        .stroke(egui::Stroke::new(1.0, LINE))
        .corner_radius(egui::CornerRadius::same(3))
        .inner_margin(egui::Margin { left: 4, right: 4, top: 1, bottom: 1 })
        .show(ui, |ui| {
            ui.label(egui::RichText::new(text).monospace().size(9.5).color(MUTED));
        });
}

/// Soft accent glow used for primary marks (e.g. brand square, active dot).
pub fn accent_glow(painter: &egui::Painter, center: egui::Pos2, radius: f32) {
    // 3 concentric circles with decreasing alpha to fake a soft bloom.
    for (mul, alpha) in [(2.6, 18), (1.8, 36), (1.2, 70)] {
        let a = egui::Color32::from_rgba_premultiplied(
            (ACCENT.r() as u16 * alpha as u16 / 255) as u8,
            (ACCENT.g() as u16 * alpha as u16 / 255) as u8,
            (ACCENT.b() as u16 * alpha as u16 / 255) as u8,
            alpha,
        );
        painter.circle_filled(center, radius * mul, a);
    }
}
