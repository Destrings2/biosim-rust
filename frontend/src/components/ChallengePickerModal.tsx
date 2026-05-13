// Gallery-style challenge picker with side detail pane and live param form.
// When the "Custom" tile is selected, the modal swaps to an editor-dominant
// layout so the JS code editor gets full width and height.

import { useEffect, useMemo, useState } from "react";
import type { Simulator } from "../../pkg/biosim4_wasm";
import type { ChallengeComposition, ChallengeConfig, ChallengeSchema } from "../types";
import { ChallengeArt, challengeArtKind } from "./ChallengeArt";
import { CustomChallengeEditor } from "./CustomChallengeEditor";
import { IcSearch } from "./Icons";
import { Modal } from "./Modal";

const CUSTOM_ID = "__custom__";

interface SchemaProperty {
  type: "number" | "boolean" | "string";
  minimum?: number;
  maximum?: number;
  default?: number | boolean | string;
  description?: string;
}
function schemaPropsOf(schema: Record<string, unknown>): Record<string, SchemaProperty> {
  return (schema.properties ?? {}) as Record<string, SchemaProperty>;
}
function defaultsFor(schema: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(schemaPropsOf(schema))) {
    if (v.default !== undefined) out[k] = v.default;
  }
  return out;
}

interface Props {
  schemas: ChallengeSchema[];
  activeId: string | null;
  simulator: Simulator;
  onClose: () => void;
  onApply: (cfg: ChallengeConfig) => void;
  onCustomRegistered: () => void;
}

export function ChallengePickerModal({ schemas, activeId, simulator, onClose, onApply, onCustomRegistered }: Props) {
  const [selectedId, setSelectedId] = useState<string>(activeId ?? schemas[0]?.id ?? "");
  const [composition, setComposition] = useState<ChallengeComposition>("Any");
  const [params, setParams] = useState<Record<string, unknown>>({});
  const [query, setQuery] = useState("");

  const isCustom = selectedId === CUSTOM_ID;

  const selected = useMemo(
    () => schemas.find((s) => s.id === selectedId) ?? null,
    [schemas, selectedId],
  );

  useEffect(() => {
    if (selected) setParams(defaultsFor(selected.schema));
  }, [selected]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      else if (e.key === "Enter" && selected && !isCustom) {
        onApply({
          active: [selected.id],
          composition,
          params: { [selected.id]: params },
        });
        onClose();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, onApply, selected, composition, params, isCustom]);

  const list = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return schemas;
    return schemas.filter((c) =>
      c.name.toLowerCase().includes(q)
      || c.id.includes(q)
      || c.description.toLowerCase().includes(q),
    );
  }, [schemas, query]);

  // ── Editor mode ─────────────────────────────────────────────────────────
  if (isCustom) {
    return (
      <Modal>
        <div className="modal-back" onClick={onClose}>
          <div className="modal picker-modal editor-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <button className="modal-close" onClick={() => setSelectedId(schemas[0]?.id ?? "")} title="Back to gallery">
                  ← GALLERY
                </button>
                <div>
                  <div className="modal-eyebrow">Custom challenge · JavaScript</div>
                  <h2 className="modal-title">Author a challenge</h2>
                </div>
              </div>
              <button className="modal-close" onClick={onClose}>ESC</button>
            </div>

            <div className="editor-body">
              <CustomChallengeEditor
                simulator={simulator}
                onApply={(id) => {
                  onCustomRegistered();
                  onApply({ active: [id], composition, params: {} });
                  onClose();
                }}
                onCancel={onClose}
              />
            </div>
          </div>
        </div>
      </Modal>
    );
  }

  // ── Gallery mode ────────────────────────────────────────────────────────
  return (
    <Modal>
    <div className="modal-back" onClick={onClose}>
      <div className="modal picker-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div>
            <div className="modal-eyebrow">Selection · {schemas.length} challenges</div>
            <h2 className="modal-title">Choose a challenge</h2>
          </div>
          <button className="modal-close" onClick={onClose}>ESC</button>
        </div>

        <div className="picker-toolbar">
          <div className="search">
            <IcSearch size={14}/>
            <input
              autoFocus
              placeholder="Search challenges, e.g. 'circle', 'sun', 'wall'…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <span className="search-kbd">/</span>
          </div>
        </div>

        <div className="picker-body">
          <div className="picker-grid">
            <button
              key={CUSTOM_ID}
              className={`ctile ctile-custom ${isCustom ? "sel" : ""}`}
              onClick={() => setSelectedId(CUSTOM_ID)}
            >
              <div className="ctile-art ctile-art-custom">
                <CustomArt />
              </div>
              <div>
                <div className="ctile-name">✎ Custom (JS)</div>
              </div>
              <div className="ctile-id">{CUSTOM_ID}</div>
              <div className="ctile-desc">Author your own challenge in JavaScript.</div>
            </button>
            {list.map((c) => (
              <button
                key={c.id}
                className={`ctile ${c.id === selectedId ? "sel" : ""}`}
                onClick={() => setSelectedId(c.id)}
              >
                <div className="ctile-art">
                  <ChallengeArt kind={challengeArtKind(c.id)} />
                </div>
                <div>
                  <div className="ctile-name">{c.name}</div>
                </div>
                <div className="ctile-id">{c.id}</div>
                <div className="ctile-desc">{c.description}</div>
              </button>
            ))}
          </div>

          <aside className="picker-side">
            <h4>Selected</h4>
            <h3 className="name">{selected?.name ?? "—"}</h3>
            <div className="id">{selected?.id ?? ""}</div>
            <p className="desc">{selected?.description ?? ""}</p>

            {selected && (
              <>
                <h4>Parameters</h4>
                {Object.entries(schemaPropsOf(selected.schema)).map(([key, prop]) => (
                  <ParamRow key={key} name={key} prop={prop} value={params[key]}
                            onChange={(v) => setParams((p) => ({ ...p, [key]: v }))} />
                ))}
                {Object.keys(schemaPropsOf(selected.schema)).length === 0 && (
                  <p style={{ color: "var(--muted)", fontSize: 11.5, fontStyle: "italic" }}>
                    No parameters.
                  </p>
                )}
              </>
            )}

            <h4 style={{ marginTop: 18 }}>Composition</h4>
            <div className="seg" style={{ width: "100%" }}>
              <button className={composition === "Any" ? "on" : ""} onClick={() => setComposition("Any")}>ANY</button>
              <button className={composition === "All" ? "on" : ""} onClick={() => setComposition("All")}>ALL</button>
            </div>
          </aside>
        </div>

        <div className="picker-foot">
          <div className="picker-comp">
            <span>↵ apply</span>
            <span>esc close</span>
          </div>
          <div style={{ display: "flex", gap: 10 }}>
            <button className="btn-ghost" onClick={onClose}>Cancel</button>
            <button
              className="btn-primary"
              disabled={!selected}
              onClick={() => {
                if (!selected) return;
                onApply({
                  active: [selected.id],
                  composition,
                  params: { [selected.id]: params },
                });
                onClose();
              }}
            >
              Apply &amp; reset epoch
            </button>
          </div>
        </div>
      </div>
    </div>
    </Modal>
  );
}

function ParamRow({ name, prop, value, onChange }: {
  name: string; prop: SchemaProperty; value: unknown;
  onChange: (v: unknown) => void;
}) {
  if (prop.type === "boolean") {
    return (
      <label style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "6px 0", borderBottom: "1px solid var(--line)", fontSize: 11, color: "var(--text-2)" }}>
        <span style={{ fontFamily: "var(--mono)" }}>{name}</span>
        <input
          type="checkbox" className="check"
          checked={Boolean(value)}
          onChange={(e) => onChange(e.target.checked)}
        />
      </label>
    );
  }
  if (prop.type === "number") {
    const min = prop.minimum ?? 0;
    const max = prop.maximum ?? 1;
    const num = typeof value === "number" ? value : Number(prop.default ?? 0);
    return (
      <div style={{ padding: "6px 0", borderBottom: "1px solid var(--line)" }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, fontFamily: "var(--mono)" }}>
          <span style={{ color: "var(--text-2)" }}>{name}</span>
          <span style={{ color: "var(--text)" }}>{num.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min={min} max={max} step={(max - min) / 100}
          value={num}
          onChange={(e) => onChange(Number(e.target.value))}
          style={{ width: "100%", marginTop: 4 }}
        />
      </div>
    );
  }
  return null;
}

// Distinctive art for the Custom tile so it doesn't read as "no art".
function CustomArt() {
  return (
    <svg viewBox="0 0 100 62" preserveAspectRatio="xMidYMid meet">
      <rect x="10" y="14" width="80" height="34" rx="3"
            fill="rgba(125,211,168,0.06)" stroke="rgba(125,211,168,0.55)" />
      <text x="50" y="36" textAnchor="middle"
            fontFamily="ui-monospace, Menlo, monospace"
            fontSize="11" fill="rgba(125,211,168,0.85)">
        {`{ } => fitness`}
      </text>
      <circle cx="18" cy="22" r="1.2" fill="rgba(125,211,168,0.5)" />
      <circle cx="22" cy="22" r="1.2" fill="rgba(125,211,168,0.5)" />
      <circle cx="26" cy="22" r="1.2" fill="rgba(125,211,168,0.5)" />
    </svg>
  );
}
