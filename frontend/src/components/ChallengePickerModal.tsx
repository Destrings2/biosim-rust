// Gallery-style challenge picker with side detail pane and live param form.

import { useEffect, useMemo, useState } from "react";
import type { ChallengeComposition, ChallengeConfig, ChallengeSchema } from "../types";
import { ChallengeArt, challengeArtKind } from "./ChallengeArt";
import { IcSearch } from "./Icons";
import { Modal } from "./Modal";

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
  onClose: () => void;
  onApply: (cfg: ChallengeConfig) => void;
}

export function ChallengePickerModal({ schemas, activeId, onClose, onApply }: Props) {
  const [selectedId, setSelectedId] = useState<string>(activeId ?? schemas[0]?.id ?? "");
  const [composition, setComposition] = useState<ChallengeComposition>("Any");
  const [params, setParams] = useState<Record<string, unknown>>({});
  const [query, setQuery] = useState("");

  const selected = useMemo(
    () => schemas.find((s) => s.id === selectedId) ?? schemas[0] ?? null,
    [schemas, selectedId],
  );

  // Reset params when the selection changes
  useEffect(() => {
    if (selected) setParams(defaultsFor(selected.schema));
  }, [selected]);

  // Esc closes
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      else if (e.key === "Enter" && selected) {
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
  }, [onClose, onApply, selected, composition, params]);

  const list = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return schemas;
    return schemas.filter((c) =>
      c.name.toLowerCase().includes(q)
      || c.id.includes(q)
      || c.description.toLowerCase().includes(q),
    );
  }, [schemas, query]);

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
