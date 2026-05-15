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

    // Tighten the default spacing to feel like a dense pro tool. Slightly
    // more vertical breathing room than horizontal so list rows / sections
    // read cleanly inside the right panel.
    let mut style: egui::Style = (*ctx.style()).clone();
    style.spacing.item_spacing = egui::vec2(8.0, 7.0);
    style.spacing.button_padding = egui::vec2(8.0, 4.0);
    style.spacing.window_margin = egui::Margin::symmetric(12, 10);
    style.spacing.indent = 14.0;
    style.spacing.slider_width = 140.0;
    // Slightly tighter scrollbar so it doesn't visually crowd the right edge.
    style.spacing.scroll.bar_width = 8.0;
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

/// Painted icon kinds — drawn from primitives rather than font glyphs so we
/// don't rely on the egui default font covering Unicode "Geometric Shapes"
/// (it doesn't — `☠ ▣ ◎ ✕ ✦ ▾ ●` all render as tofu boxes).
#[derive(Copy, Clone)]
#[allow(dead_code)] // `TabBreeds` is forward-looking — wired up in the Breeds tab work.
pub enum Icon {
    // Transport
    Play,
    Pause,
    // Chevrons
    ChevDown,
    ChevRight,
    // Status / chips
    Dot,
    // Tools
    Inspect,     // crosshair / target
    Barrier,     // filled square
    KillBarrier, // square with diagonal slash
    Kill,        // X
    Reproduce,   // 4-point spark
    // Right-panel vertical tabs
    TabStats,     // bar chart
    TabChallenge, // target rings
    TabRegistry,  // stacked lines
    TabConfig,    // sliders
    TabBreeds,    // 3-dot cluster (forward-looking)
}

/// Paint a vector icon centered in `rect` at the given stroke color.
pub fn paint_icon(painter: &egui::Painter, rect: egui::Rect, kind: Icon, color: egui::Color32) {
    let c = rect.center();
    let s = rect.width().min(rect.height());
    let stroke = egui::Stroke::new((s / 14.0).max(1.2), color);
    use egui::pos2;
    match kind {
        Icon::Play => {
            // Filled right-pointing triangle.
            let r = s * 0.30;
            let pts = vec![
                pos2(c.x - r * 0.85, c.y - r),
                pos2(c.x - r * 0.85, c.y + r),
                pos2(c.x + r * 0.95, c.y),
            ];
            painter.add(egui::Shape::convex_polygon(pts, color, egui::Stroke::NONE));
        }
        Icon::Pause => {
            let bar_w = s * 0.16;
            let bar_h = s * 0.55;
            let gap = s * 0.10;
            for sign in [-1.0_f32, 1.0] {
                let cx = c.x + sign * (bar_w * 0.5 + gap * 0.5);
                let r = egui::Rect::from_center_size(pos2(cx, c.y), egui::vec2(bar_w, bar_h));
                painter.rect_filled(r, 1.0, color);
            }
        }
        Icon::ChevDown => {
            let r = s * 0.22;
            painter.line_segment([pos2(c.x - r, c.y - r * 0.4), pos2(c.x, c.y + r * 0.5)], stroke);
            painter.line_segment([pos2(c.x, c.y + r * 0.5), pos2(c.x + r, c.y - r * 0.4)], stroke);
        }
        Icon::ChevRight => {
            let r = s * 0.22;
            painter.line_segment([pos2(c.x - r * 0.4, c.y - r), pos2(c.x + r * 0.5, c.y)], stroke);
            painter.line_segment([pos2(c.x + r * 0.5, c.y), pos2(c.x - r * 0.4, c.y + r)], stroke);
        }
        Icon::Dot => {
            painter.circle_filled(c, s * 0.16, color);
        }
        Icon::Inspect => {
            // Crosshair: circle + 4 ticks.
            let r = s * 0.28;
            painter.circle_stroke(c, r, stroke);
            let t = s * 0.10;
            painter.line_segment([pos2(c.x, c.y - r - t), pos2(c.x, c.y - r)], stroke);
            painter.line_segment([pos2(c.x, c.y + r), pos2(c.x, c.y + r + t)], stroke);
            painter.line_segment([pos2(c.x - r - t, c.y), pos2(c.x - r, c.y)], stroke);
            painter.line_segment([pos2(c.x + r, c.y), pos2(c.x + r + t, c.y)], stroke);
            painter.circle_filled(c, s * 0.06, color);
        }
        Icon::Barrier => {
            let r = s * 0.30;
            let rect = egui::Rect::from_center_size(c, egui::vec2(r * 2.0, r * 2.0));
            painter.rect(
                rect,
                egui::CornerRadius::same(1),
                color.linear_multiply(0.55),
                stroke,
                egui::StrokeKind::Inside,
            );
        }
        Icon::KillBarrier => {
            let r = s * 0.30;
            let rect = egui::Rect::from_center_size(c, egui::vec2(r * 2.0, r * 2.0));
            painter.rect_stroke(
                rect,
                egui::CornerRadius::same(1),
                stroke,
                egui::StrokeKind::Inside,
            );
            painter.line_segment(
                [pos2(c.x - r * 0.7, c.y - r * 0.7), pos2(c.x + r * 0.7, c.y + r * 0.7)],
                stroke,
            );
            painter.line_segment(
                [pos2(c.x - r * 0.7, c.y + r * 0.7), pos2(c.x + r * 0.7, c.y - r * 0.7)],
                stroke,
            );
        }
        Icon::Kill => {
            let r = s * 0.26;
            painter.line_segment([pos2(c.x - r, c.y - r), pos2(c.x + r, c.y + r)], stroke);
            painter.line_segment([pos2(c.x - r, c.y + r), pos2(c.x + r, c.y - r)], stroke);
        }
        Icon::Reproduce => {
            // 4-point spark / star.
            let r = s * 0.30;
            painter.line_segment([pos2(c.x, c.y - r), pos2(c.x, c.y + r)], stroke);
            painter.line_segment([pos2(c.x - r, c.y), pos2(c.x + r, c.y)], stroke);
            let d = r * 0.55;
            painter.line_segment([pos2(c.x - d, c.y - d), pos2(c.x + d, c.y + d)], stroke);
            painter.line_segment([pos2(c.x - d, c.y + d), pos2(c.x + d, c.y - d)], stroke);
        }
        Icon::TabStats => {
            // Three rising bars.
            let bar_w = s * 0.14;
            let gap = s * 0.07;
            let base_y = c.y + s * 0.28;
            let heights = [s * 0.20, s * 0.36, s * 0.50];
            for (i, h) in heights.iter().enumerate() {
                let cx = c.x + (i as f32 - 1.0) * (bar_w + gap);
                let r = egui::Rect::from_min_max(
                    pos2(cx - bar_w * 0.5, base_y - *h),
                    pos2(cx + bar_w * 0.5, base_y),
                );
                painter.rect_filled(r, 1.0, color);
            }
        }
        Icon::TabChallenge => {
            // Concentric rings + center dot (target).
            painter.circle_stroke(c, s * 0.34, stroke);
            painter.circle_stroke(c, s * 0.20, stroke);
            painter.circle_filled(c, s * 0.07, color);
        }
        Icon::TabRegistry => {
            // Three rows with leading bullet.
            let row_h = s * 0.18;
            let line_w = s * 0.46;
            let bullet_r = s * 0.05;
            for i in -1..=1 {
                let y = c.y + i as f32 * row_h;
                painter.circle_filled(pos2(c.x - line_w * 0.55, y), bullet_r, color);
                painter.line_segment(
                    [pos2(c.x - line_w * 0.40, y), pos2(c.x + line_w * 0.45, y)],
                    stroke,
                );
            }
        }
        Icon::TabConfig => {
            // Two horizontal "slider" tracks with thumbs at different positions.
            let row_h = s * 0.24;
            for (row, thumb_t) in [(-1, 0.30_f32), (1, 0.65)] {
                let y = c.y + row as f32 * row_h * 0.5;
                let x0 = c.x - s * 0.32;
                let x1 = c.x + s * 0.32;
                painter.line_segment([pos2(x0, y), pos2(x1, y)], stroke);
                let tx = x0 + (x1 - x0) * thumb_t;
                painter.circle_filled(pos2(tx, y), s * 0.07, color);
            }
        }
        Icon::TabBreeds => {
            // Triangle-of-three dots (forward-looking icon for the Breeds tab).
            let r = s * 0.08;
            let d = s * 0.22;
            painter.circle_filled(pos2(c.x, c.y - d), r, color);
            painter.circle_filled(pos2(c.x - d * 0.9, c.y + d * 0.6), r, color);
            painter.circle_filled(pos2(c.x + d * 0.9, c.y + d * 0.6), r, color);
        }
    }
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
