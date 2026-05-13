// Biosim dark Monaco theme. Maps the design-system palette
// (--bg, --accent, --sensor, --neuron, --warn …) onto JS/TS syntax tokens.
//
// Token assignments are semantic, not aesthetic:
//   keyword   → accent      (this is the brand colour — code "feels biosim")
//   string    → warn        (warm, easy to spot)
//   number    → sensor      (cool blue, distinct from strings)
//   constant  → neuron      (true/false/null/undefined — language constants)
//   function  → sensor      (call sites read like signals)
//   comment   → muted       (recedes)

import type { Monaco } from "@monaco-editor/react";

const PALETTE = {
  bg:     "#0a0b0d",
  bg2:    "#0f1114",
  panel:  "#14161a",
  line:   "#23272e",
  line2:  "#2d323b",
  text:   "#e6e8ec",
  text2:  "#a8aeb9",
  muted:  "#6b717c",
  accent: "#7dd3a8",
  warn:   "#e8a87c",
  bad:    "#e07b7b",
  sensor: "#6ea8e0",
  action: "#e0a86e",
  neuron: "#b08fe0",
} as const;

export const BIOSIM_THEME_ID = "biosim-dark";

// Monaco's `colors` keys must be 6- or 8-digit hex (no `rgba()`). Helper to
// convert an alpha-on-hex to the 8-digit form Monaco expects.
function withAlpha(hex: string, alpha: number): string {
  const a = Math.round(Math.max(0, Math.min(1, alpha)) * 255)
    .toString(16)
    .padStart(2, "0");
  return `${hex}${a}`;
}

export function defineBiosimTheme(monaco: Monaco): void {
  monaco.editor.defineTheme(BIOSIM_THEME_ID, {
    base: "vs-dark",
    inherit: true,
    rules: [
      { token: "",                            foreground: PALETTE.text.slice(1) },

      { token: "comment",                     foreground: PALETTE.muted.slice(1), fontStyle: "italic" },
      { token: "comment.doc",                 foreground: PALETTE.muted.slice(1), fontStyle: "italic" },

      { token: "keyword",                     foreground: PALETTE.accent.slice(1) },
      { token: "keyword.json",                foreground: PALETTE.accent.slice(1) },
      { token: "keyword.flow",                foreground: PALETTE.accent.slice(1) },
      { token: "keyword.operator",            foreground: PALETTE.text2.slice(1) },
      { token: "keyword.operator.new",        foreground: PALETTE.accent.slice(1) },

      { token: "string",                      foreground: PALETTE.warn.slice(1) },
      { token: "string.escape",               foreground: PALETTE.action.slice(1) },
      { token: "string.regexp",               foreground: PALETTE.action.slice(1) },

      { token: "number",                      foreground: PALETTE.sensor.slice(1) },
      { token: "number.hex",                  foreground: PALETTE.sensor.slice(1) },
      { token: "number.float",                foreground: PALETTE.sensor.slice(1) },

      { token: "constant",                    foreground: PALETTE.neuron.slice(1) },
      { token: "constant.language",           foreground: PALETTE.neuron.slice(1) },
      { token: "constant.numeric",            foreground: PALETTE.sensor.slice(1) },

      { token: "type",                        foreground: PALETTE.neuron.slice(1) },
      { token: "type.identifier",             foreground: PALETTE.neuron.slice(1) },
      { token: "interface",                   foreground: PALETTE.neuron.slice(1) },

      // JS identifiers — function calls and method calls in cool blue.
      { token: "identifier",                  foreground: PALETTE.text.slice(1) },
      { token: "entity.name.function",        foreground: PALETTE.sensor.slice(1) },
      { token: "support.function",            foreground: PALETTE.sensor.slice(1) },
      { token: "variable.predefined",         foreground: PALETTE.neuron.slice(1) },
      { token: "variable.parameter",          foreground: PALETTE.text2.slice(1) },
      { token: "variable.language",           foreground: PALETTE.neuron.slice(1) },

      { token: "delimiter",                   foreground: PALETTE.text2.slice(1) },
      { token: "delimiter.bracket",           foreground: PALETTE.text2.slice(1) },
      { token: "delimiter.parenthesis",       foreground: PALETTE.text2.slice(1) },
      { token: "delimiter.square",            foreground: PALETTE.text2.slice(1) },
      { token: "delimiter.curly",             foreground: PALETTE.text2.slice(1) },

      { token: "operator",                    foreground: PALETTE.text2.slice(1) },
      { token: "tag",                         foreground: PALETTE.sensor.slice(1) },
      { token: "attribute.name",              foreground: PALETTE.warn.slice(1) },
      { token: "attribute.value",             foreground: PALETTE.warn.slice(1) },

      { token: "invalid",                     foreground: PALETTE.bad.slice(1) },
    ],
    colors: {
      "editor.background":                      PALETTE.bg2,
      "editor.foreground":                      PALETTE.text,
      "editorLineNumber.foreground":            PALETTE.muted,
      "editorLineNumber.activeForeground":      PALETTE.text2,
      "editorCursor.foreground":                PALETTE.accent,
      "editor.selectionBackground":             withAlpha(PALETTE.accent, 0.22),
      "editor.selectionHighlightBackground":    withAlpha(PALETTE.accent, 0.10),
      "editor.inactiveSelectionBackground":     withAlpha(PALETTE.accent, 0.10),
      "editor.wordHighlightBackground":         withAlpha(PALETTE.accent, 0.08),
      "editor.wordHighlightStrongBackground":   withAlpha(PALETTE.accent, 0.14),
      "editor.findMatchBackground":             withAlpha(PALETTE.warn, 0.30),
      "editor.findMatchHighlightBackground":    withAlpha(PALETTE.warn, 0.15),
      "editor.lineHighlightBackground":         withAlpha(PALETTE.accent, 0.04),
      "editor.lineHighlightBorder":             "#00000000",
      "editorWhitespace.foreground":            PALETTE.line2,
      "editorIndentGuide.background1":          PALETTE.panel,
      "editorIndentGuide.activeBackground1":    PALETTE.line2,
      "editorBracketMatch.background":          withAlpha(PALETTE.accent, 0.12),
      "editorBracketMatch.border":              withAlpha(PALETTE.accent, 0.55),
      "editorRuler.foreground":                 PALETTE.line,
      "editorGutter.background":                PALETTE.bg,
      "editorGutter.modifiedBackground":        PALETTE.warn,
      "editorGutter.addedBackground":           PALETTE.accent,
      "editorGutter.deletedBackground":         PALETTE.bad,

      "editorError.foreground":                 PALETTE.bad,
      "editorWarning.foreground":               PALETTE.warn,
      "editorInfo.foreground":                  PALETTE.sensor,
      "editorHint.foreground":                  PALETTE.muted,

      "editorWidget.background":                PALETTE.panel,
      "editorWidget.border":                    PALETTE.line2,
      "editorWidget.foreground":                PALETTE.text,
      "editorSuggestWidget.background":         PALETTE.panel,
      "editorSuggestWidget.border":             PALETTE.line2,
      "editorSuggestWidget.foreground":         PALETTE.text,
      "editorSuggestWidget.selectedBackground": withAlpha(PALETTE.accent, 0.15),
      "editorSuggestWidget.highlightForeground": PALETTE.accent,
      "editorHoverWidget.background":           PALETTE.panel,
      "editorHoverWidget.border":               PALETTE.line2,
      "editorHoverWidget.foreground":           PALETTE.text,

      "scrollbar.shadow":                       "#00000000",
      "scrollbarSlider.background":             withAlpha(PALETTE.text2, 0.10),
      "scrollbarSlider.hoverBackground":        withAlpha(PALETTE.text2, 0.20),
      "scrollbarSlider.activeBackground":       withAlpha(PALETTE.text2, 0.30),

      "minimap.background":                     PALETTE.bg2,
      "minimapSlider.background":               withAlpha(PALETTE.text2, 0.10),
      "minimapSlider.hoverBackground":          withAlpha(PALETTE.text2, 0.20),
      "minimapSlider.activeBackground":         withAlpha(PALETTE.text2, 0.30),

      "diffEditor.insertedTextBackground":      withAlpha(PALETTE.accent, 0.15),
      "diffEditor.removedTextBackground":       withAlpha(PALETTE.bad, 0.15),
    },
  });
}
