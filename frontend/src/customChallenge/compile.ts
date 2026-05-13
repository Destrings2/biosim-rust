// Turn editor source into a live JS object. The source is expected to be a
// single expression (typically an object literal); we wrap it in `return (…)`
// so the user can write `({...})` directly without extra ceremony.

export interface CompileResult {
  ok: boolean;
  value?: Record<string, unknown>;
  error?: string;
}

export function compileChallenge(source: string): CompileResult {
  // Strip the optional /// <reference …/> sentinel — it's a TS hint, not runtime.
  const cleaned = source.replace(/^\s*\/\/\/\s*<reference[^>]*>\s*$/m, "");
  try {
    const fn = new Function(`"use strict"; return (${cleaned});`);
    const value = fn();
    if (!value || typeof value !== "object") {
      return { ok: false, error: "Expression must evaluate to an object literal." };
    }
    const v = value as Record<string, unknown>;
    if (typeof v.id !== "string" || !v.id) {
      return { ok: false, error: "Challenge `id` is required and must be a non-empty string." };
    }
    if (typeof v.evaluate !== "function") {
      return { ok: false, error: "Challenge `evaluate(agent, world)` is required." };
    }
    return { ok: true, value: v };
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : String(e) };
  }
}
