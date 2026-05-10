// Photoshop-style tool palette pinned above the canvas.

import { IcInspect, IcBarrier, IcKill, IcReproduce } from "./Icons";

export type Tool = "inspect" | "barrier" | "kill" | "reproduce";

interface Props {
  active: Tool;
  onChange: (t: Tool) => void;
}

const TOOLS: { id: Tool; label: string; key: string; Ic: (p: { size?: number }) => JSX.Element }[] = [
  { id: "inspect",   label: "Inspect",   key: "I", Ic: IcInspect },
  { id: "barrier",   label: "Barrier",   key: "B", Ic: IcBarrier },
  { id: "kill",      label: "Kill",      key: "K", Ic: IcKill },
  { id: "reproduce", label: "Reproduce", key: "R", Ic: IcReproduce },
];

export function FloatingToolbar({ active, onChange }: Props) {
  return (
    <div className="floating-tools">
      {TOOLS.map((t, i) => (
        <span key={t.id} style={{ display: "inline-flex" }}>
          {i === 2 && <span className="ftool-divider"/>}
          <button
            className={`ftool ${active === t.id ? "active" : ""}`}
            onClick={() => onChange(t.id)}
            title={t.label}
          >
            <t.Ic size={14}/>
            <span>{t.label}</span>
            <span className="ftool-kbd">{t.key}</span>
          </button>
        </span>
      ))}
    </div>
  );
}
