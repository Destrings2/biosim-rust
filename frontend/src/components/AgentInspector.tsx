// Three-column agent inspector: sensors+neurons list / SVG net graph /
// action outputs + genome dump. Designed as a "moment" — the SVG graph
// is the centrepiece.

import { useEffect, useMemo } from "react";
import type { Simulator } from "../../pkg/biosim4_wasm.js";
import type { NetEdge, NetNode, NetworkSnapshot } from "../types";
import { Modal } from "./Modal";

interface Props {
  agentId: number;
  simulator: Simulator;
  onClose: () => void;
}

// ── SVG layout ────────────────────────────────────────────────────────
const GRAPH_W = 720;
const GRAPH_H = 520;
const PAD = 36;

function nodesByKind(nodes: NetNode[]) {
  return {
    sensor: nodes.filter((n) => n.kind === "sensor"),
    neuron: nodes.filter((n) => n.kind === "neuron"),
    action: nodes.filter((n) => n.kind === "action"),
  };
}

function colY(count: number, h: number, pad: number): number[] {
  if (count === 0) return [];
  const usable = h - pad * 2;
  return Array.from({ length: count }, (_, i) => pad + (i + 0.5) * (usable / count));
}

const COL_X = { sensor: PAD + 130, neuron: GRAPH_W / 2, action: GRAPH_W - PAD - 20 };

export function AgentInspector({ agentId, simulator, onClose }: Props) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const net = useMemo<NetworkSnapshot | null>(() => {
    try {
      return (simulator.get_agent_network(agentId) as NetworkSnapshot | null) ?? null;
    } catch { return null; }
  }, [agentId, simulator]);

  if (!net) {
    return (
      <Modal>
      <div className="modal-back" onClick={onClose}>
        <div className="modal" onClick={(e) => e.stopPropagation()} style={{ width: 360 }}>
          <div className="modal-head">
            <div>
              <div className="modal-eyebrow">Agent #{agentId}</div>
              <h2 className="modal-title">Not found</h2>
            </div>
            <button className="modal-close" onClick={onClose}>ESC</button>
          </div>
          <div style={{ padding: 20, color: "var(--text-2)", fontSize: 12 }}>
            The agent isn't alive any more — it may have died before you clicked it.
          </div>
        </div>
      </div>
      </Modal>
    );
  }

  const { sensor: sensors, neuron: neurons, action: actions } = nodesByKind(net.nodes);
  const sY = colY(sensors.length, GRAPH_H, PAD);
  const nY = colY(neurons.length, GRAPH_H, PAD);
  const aY = colY(actions.length, GRAPH_H, PAD);

  const indexInGroup = (kind: NetNode["kind"], index: number): number => {
    const group =
      kind === "sensor" ? sensors :
      kind === "neuron" ? neurons :
                          actions;
    return group.findIndex((n) => n.index === index);
  };

  const positionOf = (kind: NetNode["kind"], index: number): { x: number; y: number } | null => {
    const i = indexInGroup(kind, index);
    if (i < 0) return null;
    const ys = kind === "sensor" ? sY : kind === "neuron" ? nY : aY;
    return { x: COL_X[kind], y: ys[i] };
  };

  const neuronOutputs = new Map(net.neuron_states.map((s) => [s.index, s.output]));

  // Compute action levels for the right column. Net edges that target an
  // action sum up to that action's pre-tanh accumulator — but we don't have
  // the simulator's true levels here, only the wiring. Instead just show
  // each action's *connection count* and the sign of its summed weights as
  // a quick "is this action driven positively or negatively" hint.
  const actionWeightSum = new Map<number, number>();
  for (const e of net.edges) {
    if (e.to_kind !== "action") continue;
    actionWeightSum.set(e.to_index, (actionWeightSum.get(e.to_index) ?? 0) + e.weight);
  }

  const swatch = `rgb(${net.color[0]},${net.color[1]},${net.color[2]})`;
  const glow = `rgb(${net.color[0]},${net.color[1]},${net.color[2]})`;

  return (
    <Modal>
    <div className="modal-back" onClick={onClose}>
      <div className="modal inspector-modal" onClick={(e) => e.stopPropagation()}>
        <div className="ins-head">
          <div className="ins-swatch" style={{ background: swatch, "--swatch-glow": `${glow}55` } as React.CSSProperties}/>
          <div>
            <div style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: "0.12em", color: "var(--muted)" }}>Agent</div>
            <div className="ins-id">#{String(net.id).padStart(4, "0")}</div>
          </div>

          <div className="ins-meta">
            <div><div className="k">Age</div><div className="v">{net.age}</div></div>
            <div><div className="k">Genome</div><div className="v">{net.genome_length}</div></div>
            <div><div className="k">Resp</div><div className="v">{net.responsiveness.toFixed(2)}</div></div>
            <div><div className="k">Neurons</div><div className="v">{neurons.length}</div></div>
            <div><div className="k">Edges</div><div className="v">{net.edges.length}</div></div>
          </div>
          <button className="modal-close" onClick={onClose}>ESC</button>
        </div>

        <div className="ins-body">
          <div className="ins-col ins-col-l">
            <h5>Sensors · {sensors.length} active</h5>
            {sensors.length === 0 && <p style={{ fontSize: 11, color: "var(--muted)", fontStyle: "italic" }}>none</p>}
            {sensors.map((n) => (
              <div key={`s${n.index}`} className="ins-li sensor">
                <span className="dot"/>
                <span className="lbl">{n.label}</span>
                <span className="v">#{n.index}</span>
              </div>
            ))}

            <h5>Neurons · {neurons.length}</h5>
            {neurons.length === 0 && <p style={{ fontSize: 11, color: "var(--muted)", fontStyle: "italic" }}>none</p>}
            {neurons.map((n) => {
              const out = neuronOutputs.get(n.index) ?? 0;
              const high = Math.abs(out) > 0.5;
              return (
                <div key={`n${n.index}`} className={`ins-li neuron ${high ? "high" : ""}`}>
                  <span className="dot"/>
                  <span className="lbl">{n.label}</span>
                  <span className="v">{out.toFixed(2)}</span>
                </div>
              );
            })}
          </div>

          <div className="ins-graph">
            <NetGraph net={net} sY={sY} nY={nY} aY={aY}
                      sensors={sensors} neurons={neurons} actions={actions}
                      positionOf={positionOf} neuronOutputs={neuronOutputs} />
          </div>

          <div className="ins-col ins-col-r">
            <h5>Actions · {actions.length} driven</h5>
            {actions.length === 0 && <p style={{ fontSize: 11, color: "var(--muted)", fontStyle: "italic" }}>none</p>}
            {actions.map((a) => {
              const sum = actionWeightSum.get(a.index) ?? 0;
              const norm = Math.max(-1, Math.min(1, sum / 4));
              const pct = Math.abs(norm) * 50;
              return (
                <div key={`a${a.index}`} className="action-bar">
                  <div className="action-bar-row">
                    <span className="action-bar-label" style={{ color: "var(--text)" }}>{a.label}</span>
                    <span className="action-bar-value">{norm >= 0 ? "+" : ""}{norm.toFixed(2)}</span>
                  </div>
                  <div className="action-bar-track">
                    <div className="action-bar-fill" style={{
                      left: norm >= 0 ? "50%" : `${50 - pct}%`,
                      width: `${pct}%`,
                      background: norm >= 0 ? "var(--accent)" : "var(--bad)",
                    }}/>
                    <div className="action-bar-mid"/>
                  </div>
                </div>
              );
            })}

            <h5 style={{ marginTop: 18 }}>Genome · {net.genome_length} genes</h5>
            <div className="genome-text">
              {/* The wasm side doesn't expose raw bytes for the genome at the
                  inspector level; show the agent id + colour as a stand-in
                  hash. Future: add an API for raw genome bytes. */}
              {`agent #${net.id} · color rgb(${net.color.join(",")}) · resp ${net.responsiveness.toFixed(3)}`}
            </div>
          </div>
        </div>
      </div>
    </div>
    </Modal>
  );
}

// ── NetGraph SVG ─────────────────────────────────────────────────────
function NetGraph({
  net, sY, nY, aY, sensors, neurons, actions, positionOf, neuronOutputs,
}: {
  net: NetworkSnapshot;
  sY: number[]; nY: number[]; aY: number[];
  sensors: NetNode[]; neurons: NetNode[]; actions: NetNode[];
  positionOf: (kind: NetNode["kind"], index: number) => { x: number; y: number } | null;
  neuronOutputs: Map<number, number>;
}) {
  return (
    <svg viewBox={`0 0 ${GRAPH_W} ${GRAPH_H}`} preserveAspectRatio="xMidYMid meet">
      <defs>
        <marker id="ai-arr-pos" viewBox="0 0 6 6" refX="6" refY="3" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M0 0 L6 3 L0 6 z" fill="#7dd3a8"/>
        </marker>
        <marker id="ai-arr-neg" viewBox="0 0 6 6" refX="6" refY="3" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M0 0 L6 3 L0 6 z" fill="#e07b7b"/>
        </marker>
      </defs>

      <text x={COL_X.sensor} y={20} textAnchor="middle" fontFamily="JetBrains Mono" fontSize="9" fill="#6b717c" letterSpacing="2">SENSORS</text>
      <text x={COL_X.neuron} y={20} textAnchor="middle" fontFamily="JetBrains Mono" fontSize="9" fill="#6b717c" letterSpacing="2">NEURONS</text>
      <text x={COL_X.action} y={20} textAnchor="middle" fontFamily="JetBrains Mono" fontSize="9" fill="#6b717c" letterSpacing="2">ACTIONS</text>

      {net.edges.map((e: NetEdge) => {
        const a = positionOf(e.from_kind, e.from_index);
        const b = positionOf(e.to_kind, e.to_index);
        if (!a || !b) return null;
        const isSelf = e.from_kind === "neuron" && e.to_kind === "neuron" && e.from_index === e.to_index;
        const edgeKey = `${e.from_kind[0]}${e.from_index}-${e.to_kind[0]}${e.to_index}`;

        let d: string;
        if (isSelf) {
          // arc above the node
          const cx = a.x + 22;
          d = `M ${a.x + 8} ${a.y - 6} Q ${cx} ${a.y - 28} ${a.x + 8} ${a.y + 6}`;
        } else {
          const mx = (a.x + b.x) / 2;
          d = `M ${a.x + 14} ${a.y} C ${mx} ${a.y} ${mx} ${b.y} ${b.x - 14} ${b.y}`;
        }
        const positive = e.weight >= 0;
        const stroke = positive ? "#7dd3a8" : "#e07b7b";
        const opacity = Math.max(0.18, Math.min(0.85, Math.abs(e.weight) / 4));
        const sw = 0.7 + Math.abs(e.weight) * 0.25;
        return (
          <path key={edgeKey} d={d} fill="none" stroke={stroke} strokeWidth={sw} opacity={opacity}
                markerEnd={`url(#${positive ? "ai-arr-pos" : "ai-arr-neg"})`} />
        );
      })}

      {sensors.map((n, i) => (
        <g key={`g-s${n.index}`} transform={`translate(${COL_X.sensor},${sY[i]})`}>
          <circle r="11" fill="#0f1114" stroke="#6ea8e0" strokeWidth="1.2"/>
          <circle r="3" fill="#6ea8e0"/>
          <text x="-18" y="-14" textAnchor="end" fontFamily="JetBrains Mono" fontSize="10" fill="#a8aeb9">{n.label}</text>
        </g>
      ))}
      {neurons.map((n, i) => {
        const out = neuronOutputs.get(n.index) ?? 0;
        const intensity = Math.min(1, Math.abs(out));
        const fillOp = 0.15 + intensity * 0.6;
        return (
          <g key={`g-n${n.index}`} transform={`translate(${COL_X.neuron},${nY[i]})`}>
            <circle r="14" fill={`rgba(176,143,224,${fillOp})`} stroke="#b08fe0" strokeWidth="1.5"/>
            <text y="3" textAnchor="middle" fontFamily="JetBrains Mono" fontSize="10" fill="#e6e8ec" fontWeight="600">{n.label}</text>
          </g>
        );
      })}
      {actions.map((n, i) => (
        <g key={`g-a${n.index}`} transform={`translate(${COL_X.action},${aY[i]})`}>
          <rect x="-12" y="-9" width="24" height="18" rx="3" fill="#0f1114" stroke="#e0a86e" strokeWidth="1.2"/>
          <circle r="2" fill="#e0a86e"/>
          <text x="-18" y="-14" textAnchor="end" fontFamily="JetBrains Mono" fontSize="10" fill="#a8aeb9">{n.label}</text>
        </g>
      ))}
    </svg>
  );
}
