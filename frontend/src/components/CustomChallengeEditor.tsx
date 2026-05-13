// Full-modal Monaco editor for user-defined challenges.
//
// Layout: editor on the left fills the modal; a narrow side rail on the right
// shows a Challenge-API quick reference and live diagnostics. Footer holds
// reset / cancel / apply.

import { useCallback, useEffect, useRef, useState } from "react";
import Editor, { type Monaco, type OnMount } from "@monaco-editor/react";
import type { Simulator } from "../../pkg/biosim4_wasm";
import { compileChallenge } from "../customChallenge/compile";
import { loadCustomSource, saveCustomSource } from "../customChallenge/storage";
import { API_DTS } from "../customChallenge/apiTypes";
import { BOILERPLATE } from "../customChallenge/template";

interface Props {
  simulator: Simulator;
  onApply: (challengeId: string) => void;
  onCancel: () => void;
}

interface Diagnostic { kind: "ok" | "error" | "info"; message: string }

export function CustomChallengeEditor({ simulator, onApply, onCancel }: Props) {
  const [source, setSource] = useState<string>(() => loadCustomSource());
  const [diag, setDiag] = useState<Diagnostic | null>({
    kind: "info",
    message: "Edit, then ⌘↵ to apply. Hover identifiers for type docs.",
  });
  const monacoRef = useRef<Monaco | null>(null);

  const handleMount: OnMount = (_editor, monaco) => {
    monacoRef.current = monaco;
    const js = monaco.languages.typescript.javascriptDefaults;
    js.setDiagnosticsOptions({
      noSemanticValidation: false,
      noSyntaxValidation: false,
      diagnosticCodesToIgnore: [
        1108, // 'return' in non-function
        2451, // Cannot redeclare block-scoped variable
      ],
    });
    js.setCompilerOptions({
      target: monaco.languages.typescript.ScriptTarget.ES2020,
      allowNonTsExtensions: true,
      allowJs: true,
      checkJs: false,
      noLib: false,
    });
    const libPath = "file:///biosim-api.d.ts";
    const existing = js.getExtraLibs?.()[libPath];
    if (!existing) js.addExtraLib(API_DTS, libPath);
  };

  const handleChange = (v: string | undefined) => {
    const next = v ?? "";
    setSource(next);
    saveCustomSource(next);
  };

  const validate = useCallback((): Record<string, unknown> | null => {
    const compiled = compileChallenge(source);
    if (!compiled.ok || !compiled.value) {
      setDiag({ kind: "error", message: compiled.error ?? "Unknown compile error." });
      return null;
    }
    try {
      const res = simulator.validate_js_challenge(compiled.value) as { ok: boolean; error?: string; id?: string };
      if (!res.ok) {
        setDiag({ kind: "error", message: res.error ?? "Validation failed." });
        return null;
      }
      setDiag({ kind: "ok", message: `Valid challenge: ${res.id}` });
      return compiled.value;
    } catch (e) {
      setDiag({ kind: "error", message: e instanceof Error ? e.message : String(e) });
      return null;
    }
  }, [source, simulator]);

  const apply = useCallback(() => {
    const v = validate();
    if (!v) return;
    try {
      const id = simulator.register_js_challenge(v) as string;
      onApply(id);
    } catch (e) {
      setDiag({ kind: "error", message: `register_js_challenge: ${e instanceof Error ? e.message : String(e)}` });
    }
  }, [validate, simulator, onApply]);

  const resetToBoilerplate = useCallback(() => {
    setSource(BOILERPLATE);
    saveCustomSource(BOILERPLATE);
    setDiag({ kind: "info", message: "Reset to boilerplate." });
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        apply();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [apply]);

  return (
    <div className="custom-editor">
      <div className="custom-editor-main">
        <Editor
          height="100%"
          defaultLanguage="javascript"
          theme="vs-dark"
          value={source}
          onChange={handleChange}
          onMount={handleMount}
          options={{
            fontSize: 13,
            minimap: { enabled: false },
            scrollBeyondLastLine: false,
            tabSize: 2,
            wordWrap: "on",
            renderLineHighlight: "all",
            automaticLayout: true,
            padding: { top: 12, bottom: 12 },
            scrollbar: { verticalScrollbarSize: 10, horizontalScrollbarSize: 10 },
          }}
        />
      </div>

      <aside className="custom-editor-side">
        <div className="ces-section">
          <h4>API quick reference</h4>
          <dl className="ces-api">
            <dt><code>id</code> <span className="req">required</span></dt>
            <dd>Stable string identifier.</dd>

            <dt><code>name</code></dt>
            <dd>Display name.</dd>

            <dt><code>description</code></dt>
            <dd>One-line summary.</dd>

            <dt><code>paramsSchema</code></dt>
            <dd>JSON Schema for configurable params.</dd>

            <dt><code>configure(params)</code></dt>
            <dd>Apply params to <code>this</code>.</dd>

            <dt><code>evaluate(agent, world)</code> <span className="req">required</span></dt>
            <dd>Return <code>{`{ pass, fitness }`}</code>.</dd>

            <dt><code>onSimStep(ctx)</code></dt>
            <dd>Per-step hook.</dd>

            <dt><code>onGenerationStart(ctx)</code></dt>
            <dd>Generation-boundary hook.</dd>

            <dt><code>overlays(world)</code></dt>
            <dd>Return <code>ChallengeOverlay[]</code>.</dd>
          </dl>
        </div>

        <div className="ces-section">
          <h4>Agent fields</h4>
          <div className="ces-pills">
            <span>id</span><span>x</span><span>y</span><span>heading</span>
            <span>age</span><span>alive</span><span>color</span>
            <span>responsiveness</span><span>breed_id</span><span>genome_length</span>
          </div>
        </div>

        <div className="ces-section">
          <h4>World fields</h4>
          <div className="ces-pills">
            <span>size_x</span><span>size_y</span><span>step</span>
            <span>generation</span><span>steps_per_generation</span>
          </div>
        </div>
      </aside>

      <footer className="custom-editor-foot">
        <div className="custom-editor-diag">
          {diag && (
            <span className={`diag-pill diag-${diag.kind}`}>
              <span className="diag-dot" /> {diag.message}
            </span>
          )}
        </div>

        <div className="custom-editor-actions">
          <button className="btn-ghost" onClick={resetToBoilerplate} title="Restore the boilerplate source">
            Reset
          </button>
          <button className="btn-ghost" onClick={onCancel}>Cancel</button>
          <button className="btn-ghost" onClick={validate}>Validate</button>
          <button className="btn-primary" onClick={apply}>
            Save &amp; Apply <span className="kbd">⌘↵</span>
          </button>
        </div>
      </footer>
    </div>
  );
}
