// Tiny SVG sparkline, fills under the curve, marks the last sample.

interface Props {
  data: number[];
  accent?: boolean;
}

export function Sparkline({ data, accent }: Props) {
  const w = 280, h = 56, pad = 4;
  if (data.length < 2) {
    return (
      <svg width="100%" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none"
           style={{ display: "block", background: "var(--panel)", border: "1px solid var(--line)", borderRadius: 6 }}>
        <text x={w / 2} y={h / 2 + 3} textAnchor="middle"
              fontFamily="JetBrains Mono" fontSize="9" fill="var(--muted)">
          waiting for data…
        </text>
      </svg>
    );
  }
  const min = Math.min(...data), max = Math.max(...data);
  const range = max - min || 1;
  const pts = data.map((v, i) => {
    const x = pad + (i / (data.length - 1)) * (w - pad * 2);
    const y = pad + (1 - (v - min) / range) * (h - pad * 2);
    return [x, y];
  });
  const d = pts.map((p, i) => `${i === 0 ? "M" : "L"} ${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(" ");
  const last = pts[pts.length - 1];
  const fillD = `${d} L ${last[0]} ${h - pad} L ${pts[0][0]} ${h - pad} Z`;
  const c = accent ? "var(--accent)" : "var(--text-2)";
  return (
    <svg width="100%" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none"
         style={{ display: "block", background: "var(--panel)", border: "1px solid var(--line)", borderRadius: 6 }}>
      <path d={fillD} fill={accent ? "var(--accent-soft)" : "rgba(168,174,185,0.08)"} />
      <path d={d} fill="none" stroke={c} strokeWidth="1.4"/>
      <circle cx={last[0]} cy={last[1]} r="2.5" fill={c}/>
    </svg>
  );
}
