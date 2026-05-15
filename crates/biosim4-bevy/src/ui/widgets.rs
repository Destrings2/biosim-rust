//! Reusable egui widgets for the right-panel sub-tabs.
//!
//! Composition: every public widget is a thin specialisation of [`field_row`]
//! — a two-column row with a stacked title+hint on the left and a body
//! region on the right. The body widgets (`slider_field_*`,
//! `stepper_field`, `toggle_field`, `seed_field`, `enum_field_u8`) all
//! right-anchor their controls by allocating a fixed-width sub-Ui via
//! [`right_strip`], rather than via `with_layout(right_to_left)`. Explicit
//! allocation makes the geometry obvious at the call site and prevents
//! the slider's track from bleeding into the value strip.
//!
//! Type plumbing: the slider / chip / stepper logic is generic over the
//! numeric trait egui uses internally (`emath::Numeric`). The public
//! wrappers (`slider_field_u32`, `slider_field_u16`, `slider_field_f32`)
//! exist only so call sites can pass `1..=64` literals without turbofish.
//! Each is a one-liner over the generic core.
//!
//! These widgets are deliberately panel-agnostic: they take `&mut egui::Ui`
//! and primitive refs, so the Challenge / Breeds tabs can reuse them.

use std::ops::RangeInclusive;

use bevy_egui::egui;
use egui::emath::Numeric;

use crate::theme;

// ─── Grid geometry ──────────────────────────────────────────────────────────
//
// The right panel is 360px wide. After the 38px tab strip, 14px+14px body
// inset, and SCROLLBAR_RESERVE for the vertical scrollbar, ~280px of usable
// width remains. Split into a 110px label column + 6px gap + ~164px body.

/// Vertical gap inserted after each field row.
pub const FIELD_GAP: f32 = 4.0;
/// Width of the left label column. Titles longer than this ellipsize via
/// `Label::truncate()`; the original text is surfaced on hover.
pub const LABEL_COL_WIDTH: f32 = 110.0;
/// Horizontal gap between the label and body columns.
const COL_GAP: f32 = 6.0;
/// Reserve for the vertical scrollbar. egui's `available_width()` inside a
/// `ScrollArea` reports the full scrollable width regardless of whether the
/// scrollbar is rendered on top, so without this reserve right-anchored
/// content slides under the scrollbar thumb.
const SCROLLBAR_RESERVE: f32 = 14.0;

/// Height of every body widget (slider, drag value, stepper button) so
/// adjacent rows line up to the pixel.
pub const BODY_HEIGHT: f32 = 20.0;
/// Width of the right-side value strip on slider rows. Fits "20,000" or
/// "0.0500" with a few pixels of margin from the slider's end handle.
const VALUE_STRIP_WIDTH: f32 = 56.0;
/// Gap between the slider's right edge and the value strip. Larger than the
/// default item spacing so the slider handle doesn't kiss the number.
const SLIDER_VALUE_GAP: f32 = 8.0;
/// Drag-value box inside steppers — fits 5-digit unsigned integers.
const STEPPER_VALUE_WIDTH: f32 = 44.0;
/// Side length of stepper `[−]` / `[+]` buttons.
const STEPPER_BUTTON_SIZE: f32 = 20.0;
/// Gap between `[−]` value `[+]` in a stepper.
const STEPPER_GAP: f32 = 4.0;
const TOGGLE_W: f32 = 30.0;
const TOGGLE_H: f32 = 18.0;
const SEED_VALUE_WIDTH: f32 = 80.0;
const ICON_BUTTON_SIZE: f32 = 22.0;

// ─── Section header ─────────────────────────────────────────────────────────

/// Section divider with title on the left and an optional right-aligned
/// summary (e.g. `"24 genes · 5 neurons"`).
pub fn section_header(ui: &mut egui::Ui, title: &str, tally: Option<&str>) {
    ui.add_space(8.0);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(title).size(10.0).color(theme::ACCENT).strong());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if let Some(t) = tally {
                ui.label(egui::RichText::new(t).monospace().size(10.5).color(theme::TEXT_2));
            }
        });
    });
    let avail = ui.available_width();
    let (r, _) = ui.allocate_exact_size(egui::vec2(avail, 1.0), egui::Sense::hover());
    ui.painter().rect_filled(r, 0.0, theme::LINE);
    ui.add_space(6.0);
}

// ─── Field row primitive ────────────────────────────────────────────────────

/// Two-column row: stacked title+hint on the left, body on the right.
///
/// Grid invariants enforced here:
///   1. Label column is always `LABEL_COL_WIDTH` wide (pinned via
///      `set_min_size` because `allocate_ui_with_layout`'s parent cursor
///      advances by the child's `min_rect`, which would otherwise shrink
///      to content for short titles like "Threads").
///   2. Body column is always `body_w` wide, same way.
///   3. Title and hint never wrap (`Label::truncate()`); long text
///      ellipsizes and surfaces in a hover tooltip.
///   4. Body uses `top_down_justified` so each row inside fills `body_w`.
pub fn field_row(
    ui: &mut egui::Ui,
    title: &str,
    hint: Option<&str>,
    body: impl FnOnce(&mut egui::Ui),
) {
    let total = (ui.available_width() - SCROLLBAR_RESERVE).max(60.0);
    let body_w = (total - LABEL_COL_WIDTH - COL_GAP).max(60.0);
    ui.horizontal_top(|ui| {
        ui.allocate_ui_with_layout(
            egui::vec2(LABEL_COL_WIDTH, 0.0),
            egui::Layout::top_down(egui::Align::Min),
            |ui| {
                ui.set_min_size(egui::vec2(LABEL_COL_WIDTH, 1.0));
                ui.add_space(2.0);
                let title_resp = ui.add(
                    egui::Label::new(egui::RichText::new(title).size(11.5).color(theme::TEXT))
                        .truncate(),
                );
                if let Some(h) = hint {
                    let hint_resp = ui.add(
                        egui::Label::new(egui::RichText::new(h).size(10.0).color(theme::MUTED))
                            .truncate(),
                    );
                    let _ = title_resp.on_hover_text(format!("{title} — {h}"));
                    let _ = hint_resp.on_hover_text(h);
                }
            },
        );
        ui.add_space(COL_GAP);
        ui.allocate_ui_with_layout(
            egui::vec2(body_w, 0.0),
            egui::Layout::top_down_justified(egui::Align::Min),
            |ui| {
                ui.set_min_size(egui::vec2(body_w, 1.0));
                body(ui);
            },
        );
    });
    ui.add_space(FIELD_GAP);
}

// ─── Slider rows ────────────────────────────────────────────────────────────

/// u32 slider + comma-formatted value strip + optional preset chip row.
pub fn slider_field_u32(
    ui: &mut egui::Ui,
    title: &str,
    hint: Option<&str>,
    value: &mut u32,
    range: RangeInclusive<u32>,
    presets: &[u32],
) {
    field_row(ui, title, hint, |ui| {
        slider_row(ui, title, value, range.clone(), &|v: u32| format_u32_commas(v));
        if !presets.is_empty() {
            ui.add_space(4.0);
            chip_row(ui, value, presets, &|v: u32| format_u32_commas(v));
        }
    });
}

/// u16 slider — same shape as the u32 variant.
pub fn slider_field_u16(
    ui: &mut egui::Ui,
    title: &str,
    hint: Option<&str>,
    value: &mut u16,
    range: RangeInclusive<u16>,
    presets: &[u16],
) {
    field_row(ui, title, hint, |ui| {
        slider_row(ui, title, value, range.clone(), &|v: u16| format_u32_commas(v as u32));
        if !presets.is_empty() {
            ui.add_space(4.0);
            chip_row(ui, value, presets, &|v: u16| format_u32_commas(v as u32));
        }
    });
}

/// f32 slider with a caller-supplied formatter.
pub fn slider_field_f32(
    ui: &mut egui::Ui,
    title: &str,
    hint: Option<&str>,
    value: &mut f32,
    range: RangeInclusive<f32>,
    fmt: impl Fn(f32) -> String,
) {
    field_row(ui, title, hint, |ui| {
        slider_row(ui, title, value, range, &fmt);
    });
}

/// Slider with a single value driving the bar, and the *current* min/max of
/// the underlying pair shown as a "MIN x   MAX y" footer. Used for the
/// GENETICS / genome length control where both bounds are interesting.
pub fn slider_with_bounds_u16(
    ui: &mut egui::Ui,
    title: &str,
    hint: Option<&str>,
    value: &mut u16,
    range: RangeInclusive<u16>,
    min_label: u16,
    max_label: u16,
) {
    field_row(ui, title, hint, |ui| {
        ui.add_sized(
            egui::vec2(ui.available_width(), BODY_HEIGHT),
            egui::Slider::new(value, range).show_value(false),
        );
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new(format!("MIN {min_label}"))
                    .monospace()
                    .size(10.0)
                    .color(theme::MUTED),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    egui::RichText::new(format!("MAX {max_label}"))
                        .monospace()
                        .size(10.0)
                        .color(theme::MUTED),
                );
            });
        });
    });
}

// ─── Stepper rows ───────────────────────────────────────────────────────────

/// `[−] value [+]` stepper. `value` and `range` plumb through `DragValue`'s
/// existing `Numeric` bound, so this one function serves u8/u16/u32/etc.
pub fn stepper_field<T: Numeric + Copy + PartialOrd>(
    ui: &mut egui::Ui,
    title: &str,
    hint: Option<&str>,
    value: &mut T,
    range: RangeInclusive<T>,
) {
    field_row(ui, title, hint, |ui| {
        let avail = ui.available_width();
        let total_w = STEPPER_BUTTON_SIZE * 2.0 + STEPPER_VALUE_WIDTH + STEPPER_GAP * 2.0;
        ui.horizontal(|ui| {
            ui.add_space((avail - total_w).max(0.0));
            let minus = stepper_button(ui, "−");
            ui.add_space(STEPPER_GAP);
            styled_drag_value(ui, value, range.clone());
            ui.add_space(STEPPER_GAP);
            let plus = stepper_button(ui, "+");
            let cur = value.to_f64();
            let min = range.start().to_f64();
            let max = range.end().to_f64();
            if plus && cur < max {
                *value = T::from_f64(cur + 1.0);
            }
            if minus && cur > min {
                *value = T::from_f64(cur - 1.0);
            }
        });
    });
}

// ─── Toggle ─────────────────────────────────────────────────────────────────

/// Pill-style switch. Mutates `on`; returns the response so callers can
/// attach hover tooltips.
pub fn toggle(ui: &mut egui::Ui, on: &mut bool) -> egui::Response {
    let (rect, mut resp) =
        ui.allocate_exact_size(egui::vec2(TOGGLE_W, TOGGLE_H), egui::Sense::click());
    if resp.clicked() {
        *on = !*on;
        resp.mark_changed();
    }
    let painter = ui.painter();
    let bg = if *on { theme::ACCENT } else { theme::LINE };
    painter.rect_filled(rect, egui::CornerRadius::same((rect.height() * 0.5) as u8), bg);
    let knob_r = (rect.height() - 4.0) * 0.5;
    let knob_x = if *on { rect.right() - knob_r - 2.0 } else { rect.left() + knob_r + 2.0 };
    painter.circle_filled(egui::pos2(knob_x, rect.center().y), knob_r, theme::TEXT);
    resp
}

/// Field row + right-anchored [`toggle`].
pub fn toggle_field(
    ui: &mut egui::Ui,
    title: &str,
    hint: Option<&str>,
    on: &mut bool,
) -> egui::Response {
    let mut out: Option<egui::Response> = None;
    field_row(ui, title, hint, |ui| {
        ui.horizontal(|ui| {
            let avail = ui.available_width();
            ui.add_space((avail - TOGGLE_W).max(0.0));
            out = Some(toggle(ui, on));
        });
    });
    out.expect("field_row always invokes its body")
}

// ─── Seed (regen icon) ──────────────────────────────────────────────────────

/// Drag-value input with a circular regen icon to its right. Returns `true`
/// the frame the user clicks regen.
pub fn seed_field(ui: &mut egui::Ui, title: &str, hint: Option<&str>, value: &mut u64) -> bool {
    let mut regen = false;
    field_row(ui, title, hint, |ui| {
        let avail = ui.available_width();
        let row_w = SEED_VALUE_WIDTH + 4.0 + ICON_BUTTON_SIZE;
        ui.horizontal(|ui| {
            ui.add_space((avail - row_w).max(0.0));
            styled_drag_value(ui, value, 0..=u64::MAX);
            ui.add_space(4.0);
            let (rect, resp) = ui.allocate_exact_size(
                egui::vec2(ICON_BUTTON_SIZE, BODY_HEIGHT),
                egui::Sense::click(),
            );
            let hovered = resp.hovered();
            let painter = ui.painter();
            painter.rect(
                rect,
                egui::CornerRadius::same(4),
                if hovered { theme::PANEL_2 } else { egui::Color32::TRANSPARENT },
                egui::Stroke::new(1.0, theme::LINE),
                egui::StrokeKind::Inside,
            );
            theme::paint_icon(
                painter,
                rect.shrink(4.0),
                theme::Icon::Refresh,
                if hovered { theme::ACCENT } else { theme::TEXT_2 },
            );
            let _ = resp.clone().on_hover_text("Randomize seed");
            if resp.clicked() {
                regen = true;
            }
        });
    });
    regen
}

// ─── Enum dropdown ──────────────────────────────────────────────────────────

/// Look up a display label by value — useful for read-only displays that
/// want to show the same text the dropdown shows collapsed.
pub fn enum_label(options: &[(u8, &'static str)], v: u8) -> &'static str {
    options.iter().find(|(val, _)| *val == v).map(|(_, name)| *name).unwrap_or("?")
}

/// Right-aligned ComboBox for a u8 field with a fixed set of `(value, label)`
/// choices.
pub fn enum_field_u8(
    ui: &mut egui::Ui,
    title: &str,
    hint: Option<&str>,
    value: &mut u8,
    options: &[(u8, &'static str)],
) {
    let current = enum_label(options, *value);
    field_row(ui, title, hint, |ui| {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            egui::ComboBox::from_id_salt(title)
                .selected_text(egui::RichText::new(current).size(11.0).color(theme::TEXT))
                .show_ui(ui, |ui| {
                    for (val, name) in options {
                        if ui.selectable_label(*value == *val, *name).clicked() {
                            *value = *val;
                        }
                    }
                });
        });
    });
}

// ─── Generic slider / chip core ─────────────────────────────────────────────

/// Slider + fixed-width value strip on the right.
///
/// `egui::Slider` allocates `ui.spacing().slider_width` regardless of what
/// `add_sized` passes — the theme sets that to 140, which would overflow
/// the body column. Overriding `slider_width` inside a `ui.scope` is the
/// only way to force a narrower slider.
///
/// `id_source` is the row's title — it makes the per-row edit-mode state
/// (kept in `egui::Memory`) collision-free without plumbing identifiers
/// through every public wrapper.
fn slider_row<T: Numeric>(
    ui: &mut egui::Ui,
    id_source: &str,
    value: &mut T,
    range: RangeInclusive<T>,
    fmt: &dyn Fn(T) -> String,
) {
    ui.horizontal(|ui| {
        let avail = ui.available_width();
        let slider_w = (avail - VALUE_STRIP_WIDTH - SLIDER_VALUE_GAP).max(60.0);
        ui.scope(|ui| {
            ui.spacing_mut().slider_width = slider_w;
            ui.add(egui::Slider::new(value, range.clone()).show_value(false));
        });
        ui.add_space(SLIDER_VALUE_GAP);
        editable_value_strip(ui, id_source, value, range, fmt);
    });
}

/// Value strip on the right of a slider row. Double-clicking the label
/// swaps it for an inline `TextEdit`; Enter or focus loss commits, Esc
/// cancels. The transient buffer lives in `egui::Memory`, keyed by
/// `id_source`, so no extra plumbing is needed at call sites.
fn editable_value_strip<T: Numeric>(
    ui: &mut egui::Ui,
    id_source: &str,
    value: &mut T,
    range: RangeInclusive<T>,
    fmt: &dyn Fn(T) -> String,
) {
    let edit_id = ui.id().with(("slider_edit", id_source));
    let textedit_id = edit_id.with("textedit");
    let mut buffer: Option<String> = ui.data(|d| d.get_temp(edit_id));

    ui.allocate_ui_with_layout(
        egui::vec2(VALUE_STRIP_WIDTH, BODY_HEIGHT),
        egui::Layout::right_to_left(egui::Align::Center),
        |ui| {
            if let Some(buf) = buffer.as_mut() {
                let resp = ui.add(
                    egui::TextEdit::singleline(buf)
                        .id(textedit_id)
                        .desired_width(VALUE_STRIP_WIDTH)
                        .font(egui::TextStyle::Monospace),
                );
                let enter = ui.input(|i| i.key_pressed(egui::Key::Enter));
                let esc = ui.input(|i| i.key_pressed(egui::Key::Escape));
                let commit = enter || (resp.lost_focus() && !esc);
                if commit {
                    if let Ok(parsed) = buf.replace(',', "").parse::<f64>() {
                        let min = range.start().to_f64();
                        let max = range.end().to_f64();
                        *value = T::from_f64(parsed.clamp(min, max));
                    }
                    ui.data_mut(|d| d.remove::<String>(edit_id));
                } else if esc {
                    ui.data_mut(|d| d.remove::<String>(edit_id));
                } else {
                    ui.data_mut(|d| d.insert_temp(edit_id, buf.clone()));
                }
            } else {
                let label =
                    egui::RichText::new(fmt(*value)).monospace().size(11.5).color(theme::TEXT);
                let resp = ui
                    .add(egui::Label::new(label).sense(egui::Sense::click()))
                    .on_hover_text("Double-click to edit");
                if resp.double_clicked() {
                    let seed = fmt(*value).replace(',', "");
                    // Pre-select the seeded text so the next keystroke
                    // replaces the value instead of appending to it.
                    let mut state =
                        egui::widgets::text_edit::TextEditState::load(ui.ctx(), textedit_id)
                            .unwrap_or_default();
                    let end = seed.chars().count();
                    state.cursor.set_char_range(Some(egui::text::CCursorRange::two(
                        egui::text::CCursor::new(0),
                        egui::text::CCursor::new(end),
                    )));
                    state.store(ui.ctx(), textedit_id);
                    ui.data_mut(|d| d.insert_temp(edit_id, seed));
                    ui.memory_mut(|m| m.request_focus(textedit_id));
                }
            }
        },
    );
}

fn chip_row<T: Numeric + Copy + PartialEq>(
    ui: &mut egui::Ui,
    value: &mut T,
    presets: &[T],
    fmt: &dyn Fn(T) -> String,
) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        let n = presets.len() as f32;
        let avail = ui.available_width();
        let chip_w = ((avail - (n - 1.0) * 4.0) / n).max(28.0);
        for &p in presets {
            if chip(ui, *value == p, &fmt(p), chip_w) {
                *value = p;
            }
        }
    });
}

// ─── Internal building blocks ───────────────────────────────────────────────

/// Custom-painted clickable rect. We can't use `egui::Button` here because
/// its desired_size is `max(text + spacing.button_padding, interact_size)`
/// — with the theme's 8px button_padding and 40px interact_size, a chip
/// labeled "5,000" wants to be 46px wide even when `add_sized` says 38.
/// Painting manually gives the exact cell width we computed.
fn chip(ui: &mut egui::Ui, active: bool, label: &str, width: f32) -> bool {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(width, 22.0), egui::Sense::click());
    let (fill, stroke_col, text_col) = if active {
        (theme::ACCENT_SOFT, theme::ACCENT, theme::ACCENT)
    } else if resp.hovered() {
        (theme::PANEL_2, theme::LINE_2, theme::TEXT)
    } else {
        (egui::Color32::TRANSPARENT, theme::LINE, theme::TEXT_2)
    };
    let painter = ui.painter();
    painter.rect(
        rect,
        egui::CornerRadius::same(4),
        fill,
        egui::Stroke::new(1.0, stroke_col),
        egui::StrokeKind::Inside,
    );
    painter.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        label,
        egui::FontId::monospace(9.5),
        text_col,
    );
    resp.clicked()
}

fn stepper_button(ui: &mut egui::Ui, label: &str) -> bool {
    let (rect, resp) =
        ui.allocate_exact_size(egui::vec2(STEPPER_BUTTON_SIZE, BODY_HEIGHT), egui::Sense::click());
    let fill = if resp.hovered() { theme::LINE } else { theme::PANEL_2 };
    let text_col = if resp.hovered() { theme::TEXT } else { theme::TEXT_2 };
    let painter = ui.painter();
    painter.rect(
        rect,
        egui::CornerRadius::same(4),
        fill,
        egui::Stroke::new(1.0, theme::LINE),
        egui::StrokeKind::Inside,
    );
    painter.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        label,
        egui::FontId::monospace(13.0),
        text_col,
    );
    resp.clicked()
}

/// `DragValue` styled to match `stepper_button` so the `[−] value [+]`
/// trio looks like one widget. Without this scope, egui's default
/// widget visuals (different fill / stroke / corner_radius) make the
/// value box stand out next to the buttons.
fn styled_drag_value<T: Numeric>(ui: &mut egui::Ui, value: &mut T, range: RangeInclusive<T>) {
    let width = if std::mem::size_of::<T>() <= 4 { STEPPER_VALUE_WIDTH } else { SEED_VALUE_WIDTH };
    ui.scope(|ui| {
        let v = ui.visuals_mut();
        for state in [
            &mut v.widgets.inactive,
            &mut v.widgets.hovered,
            &mut v.widgets.active,
            &mut v.widgets.open,
        ] {
            state.bg_fill = theme::PANEL_2;
            state.weak_bg_fill = theme::PANEL_2;
            state.bg_stroke = egui::Stroke::new(1.0, theme::LINE);
            state.corner_radius = egui::CornerRadius::same(4);
            state.fg_stroke.color = theme::TEXT;
        }
        ui.add_sized(egui::vec2(width, BODY_HEIGHT), egui::DragValue::new(value).range(range));
    });
}

/// Group thousands with commas — `1234` → `"1,234"`.
pub fn format_u32_commas(v: u32) -> String {
    let s = v.to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out.chars().rev().collect()
}
