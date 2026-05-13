// Full-modal Monaco editor for user-defined challenges.
//
// Layout: editor on the left fills the modal; a narrow tabbed side rail on the
// right shows agent/world field reference and code snippets. Footer has a
// status bar and an action bar.

import { useCallback, useEffect, useRef, useState } from "react";
import Editor, { type Monaco, type OnMount } from "@monaco-editor/react";
import type { Simulator } from "../../pkg/biosim4_wasm";
import { compileChallenge } from "../customChallenge/compile";
import { API_DTS } from "../customChallenge/apiTypes";

interface Props {
  simulator: Simulator;
  source: string;
  onSourceChange: (v: string) => void;
  onApply: (challengeId: string) => void;
  onCancel: () => void;
}

interface Diagnostic { kind: "ok" | "error" | "info"; message: string }

type SideTab = "agent" | "world" | "snippets";

const AGENT_FIELDS = [
  { name: "agent.x",             type: "NUM",     desc: "World X cell (0..size_x-1)" },
  { name: "agent.y",             type: "NUM",     desc: "World Y cell (0..size_y-1)" },
  { name: "agent.age",           type: "NUM",     desc: "Steps lived this generation" },
  { name: "agent.heading",       type: "NUM",     desc: "Compass ordinal 0..7 (N=0, NE=1, …)" },
  { name: "agent.responsiveness",type: "NUM",     desc: "Responsiveness modulator 0..1" },
  { name: "agent.color",         type: "[N,N,N]", desc: "RGB triple" },
  { name: "agent.alive",         type: "BOOL",    desc: "False if died before evaluation" },
  { name: "agent.breed_id",      type: "NUM",     desc: "Numeric breed id (default 0)" },
  { name: "agent.genome_length", type: "NUM",     desc: "Number of genes in the genome" },
  { name: "agent.id",            type: "NUM",     desc: "Stable non-zero agent id" },
] as const;

const WORLD_FIELDS = [
  { name: "world.size_x",               type: "NUM", desc: "Grid width in cells" },
  { name: "world.size_y",               type: "NUM", desc: "Grid height in cells" },
  { name: "world.generation",           type: "NUM", desc: "Current generation (0-based)" },
  { name: "world.step",                 type: "NUM", desc: "Step within generation" },
  { name: "world.steps_per_generation", type: "NUM", desc: "Steps per generation" },
] as const;

function typeBadgeClass(type: string): string {
  if (type === "BOOL") return "badge warn";
  if (type === "[N,N,N]") return "badge";
  return "badge ok";
}

export function CustomChallengeEditor({ simulator, source, onSourceChange, onApply, onCancel }: Props) {
  const [diag, setDiag] = useState<Diagnostic | null>({
    kind: "info",
    message: "Edit, then ⌘↵ to apply. Hover identifiers for type docs.",
  });
  const [sideTab, setSideTab] = useState<SideTab>("agent");
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
    onSourceChange(v ?? "");
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
        <div className="ces-tab-bar">
          {(["agent", "world", "snippets"] as SideTab[]).map((t) => (
            <button
              key={t}
              className={`ces-tab-btn${sideTab === t ? " active" : ""}`}
              onClick={() => setSideTab(t)}
            >
              {t}
            </button>
          ))}
        </div>

        {sideTab === "agent" && (
          <div className="ces-fields">
            {AGENT_FIELDS.map((f) => (
              <div key={f.name} className="ces-field">
                <div className="ces-field-head">
                  <span className="ces-field-name">{f.name}</span>
                  <span className={typeBadgeClass(f.type)}>{f.type}</span>
                </div>
                <div className="ces-field-desc">{f.desc}</div>
              </div>
            ))}
          </div>
        )}

        {sideTab === "world" && (
          <div className="ces-fields">
            {WORLD_FIELDS.map((f) => (
              <div key={f.name} className="ces-field">
                <div className="ces-field-head">
                  <span className="ces-field-name">{f.name}</span>
                  <span className="badge ok">{f.type}</span>
                </div>
                <div className="ces-field-desc">{f.desc}</div>
              </div>
            ))}
          </div>
        )}

        {sideTab === "snippets" && (
          <div className="ces-snippets">
            <div className="ces-snippet">
              <div className="ces-snippet-label">Distance to centre</div>
              <pre className="ces-snippet-code">{`const cx = world.size_x / 2;
const cy = world.size_y / 2;
const d = Math.hypot(agent.x - cx, agent.y - cy);`}</pre>
            </div>
            <div className="ces-snippet">
              <div className="ces-snippet-label">Normalised fitness</div>
              <pre className="ces-snippet-code">{`const maxR = Math.min(world.size_x, world.size_y) / 2;
const fitness = Math.max(0, 1 - d / maxR);`}</pre>
            </div>
            <div className="ces-snippet">
              <div className="ces-snippet-label">Minimal challenge shape</div>
              <pre className="ces-snippet-code">{`({
  id: "my_challenge",
  evaluate(agent, world) {
    return { pass: true, fitness: 1 };
  },
})`}</pre>
            </div>
          </div>
        )}
      </aside>

      <footer className="custom-editor-foot">
        <div className="custom-editor-status">
          <div className="custom-editor-diag">
            {diag && (
              <span className={`diag-pill diag-${diag.kind}`}>
                <span className="diag-dot" /> {diag.message}
              </span>
            )}
          </div>
          <div className="custom-editor-charcount">
            {source.length} chars · runs once per agent at generation end
          </div>
        </div>

        <div className="custom-editor-hint-bar">
          <div className="hint-keys">
            <span>⌘↵ Apply</span>
            <span className="hint-sep">·</span>
            <span>Tab Indent</span>
          </div>
          <div className="custom-editor-actions">
            <button className="btn-ghost" onClick={onCancel}>Back to gallery</button>
            <button className="btn-primary" onClick={apply}>Apply custom challenge</button>
          </div>
        </div>
      </footer>
    </div>
  );
}
