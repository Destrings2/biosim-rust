import React from "react";
import { IcChevron } from "./Icons";

export function CfgGroup({ title, summary, children }: { title: string; summary: string; children: React.ReactNode }) {
  const [open, setOpen] = React.useState(true);
  return (
    <div className={`cfg-group ${open ? "open" : ""}`}>
      <button className="cfg-group-h" onClick={()=>setOpen(o=>!o)}>
        <span className="cfg-group-chev">
          <IcChevron size={12}/>
        </span>
        <span className="cfg-group-t">{title}</span>
        <span className="cfg-group-s">{summary}</span>
      </button>
      {open && <div className="cfg-group-body">{children}</div>}
    </div>
  );
}

export function CfgRow({ label, help, children }: { label: string; help?: string; children: React.ReactNode }) {
  return (
    <div className="cfg-row">
      <div className="cfg-row-l">
        <div className="cfg-row-lbl">{label}</div>
        {help && <div className="cfg-row-help">{help}</div>}
      </div>
      <div className="cfg-row-r">{children}</div>
    </div>
  );
}

export function Stepper({ value, min, max, step, onChange, suffix }: { value: number; min: number; max: number; step: number; onChange: (v: number) => void; suffix?: string }) {
  const dec = () => onChange(Math.max(min, +(value - step).toFixed(6)));
  const inc = () => onChange(Math.min(max, +(value + step).toFixed(6)));
  return (
    <div className="stepper">
      <button onClick={dec} disabled={value <= min}>−</button>
      <input type="text" value={suffix ? `${value} ${suffix}` : value} readOnly />
      <button onClick={inc} disabled={value >= max}>+</button>
    </div>
  );
}

export function SliderNum({ value, min, max, step, onChange, format, markers, markerLabels }: {
  value: number; min: number; max: number; step: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  markers?: number[]; markerLabels?: string[];
}) {
  const display = format ? format(value) : value.toLocaleString();
  return (
    <div className="slidernum">
      <div className="slidernum-top">
        <input type="range" min={min} max={max} step={step} value={value}
               onChange={(e)=>onChange(+e.target.value)}/>
        <div className="slidernum-v">{display}</div>
      </div>
      {markers && (
        <div className="slidernum-ticks">
          {markers.map((m, i) => (
            <button key={m} className={value === m ? "on" : ""}
                    onClick={()=>onChange(m)}>
              {markerLabels ? markerLabels[i] : m.toLocaleString()}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export function RangeDual({ min, max, low, high, onChange }: {
  min: number; max: number; low: number; high: number;
  onChange: (lo: number, hi: number) => void;
}) {
  const pctLo = ((low - min) / (max - min)) * 100;
  const pctHi = ((high - min) / (max - min)) * 100;
  return (
    <div className="rangedual">
      <div className="rangedual-top">
        <div className="rangedual-track">
          <div className="rangedual-fill" style={{left: `${pctLo}%`, right: `${100-pctHi}%`}}/>
        </div>
        <input type="range" min={min} max={max} value={low}
               onChange={(e)=>{
                 const v = Math.min(+e.target.value, high);
                 onChange(v, high);
               }}/>
        <input type="range" min={min} max={max} value={high}
               onChange={(e)=>{
                 const v = Math.max(+e.target.value, low);
                 onChange(low, v);
               }}/>
      </div>
      <div className="rangedual-vals">
        <span><span className="cfg-mono-dim">min</span> <span className="cfg-mono">{low}</span></span>
        <span><span className="cfg-mono-dim">max</span> <span className="cfg-mono">{high}</span></span>
      </div>
    </div>
  );
}

export function Toggle({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <button className={`toggle ${checked ? "on" : ""}`}
            onClick={()=>onChange(!checked)}
            role="switch" aria-checked={checked}>
      <span className="toggle-knob"/>
    </button>
  );
}

export function SeedInput({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  return (
    <div className="seedinput">
      <input type="number" value={value} onChange={(e)=>onChange(+e.target.value)}/>
      <button className="seed-btn" title="Randomize"
              onClick={()=>onChange(Math.floor(Math.random() * 1e6))}>
        ⟳
      </button>
    </div>
  );
}

export function BarrierPicker({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  const opts = [
    { id: 0, label: "None",         art: <g/> },
    { id: 1, label: "Floaters",     art: <g fill="currentColor"><rect x="18" y="22" width="4" height="4"/><rect x="22" y="14" width="4" height="4"/><rect x="26" y="26" width="4" height="4"/></g> },
    { id: 2, label: "Vertical",     art: <rect x="22" y="12" width="4" height="24" fill="currentColor"/> },
    { id: 3, label: "Horizontal",   art: <rect x="12" y="22" width="24" height="4" fill="currentColor"/> },
    { id: 4, label: "Checker",      art: <g fill="currentColor"><rect x="12" y="12" width="4" height="4"/><rect x="20" y="20" width="4" height="4"/><rect x="28" y="28" width="4" height="4"/><rect x="36" y="36" width="4" height="4"/><rect x="12" y="28" width="4" height="4"/><rect x="28" y="12" width="4" height="4"/></g> },
    { id: 5, label: "Split Walls",  art: <g fill="currentColor"><rect x="14" y="6" width="4" height="14"/><rect x="14" y="28" width="4" height="14"/><rect x="30" y="6" width="4" height="14"/><rect x="30" y="28" width="4" height="14"/></g> },
    { id: 6, label: "Quincunx",     art: <g fill="currentColor"><rect x="12" y="12" width="4" height="4"/><rect x="32" y="12" width="4" height="4"/><rect x="12" y="32" width="4" height="4"/><rect x="32" y="32" width="4" height="4"/><rect x="22" y="22" width="4" height="4"/></g> },
    { id: 7, label: "Strips",       art: <g fill="currentColor"><rect x="8" y="12" width="32" height="2"/><rect x="8" y="23" width="32" height="2"/><rect x="8" y="34" width="32" height="2"/></g> },
  ];
  return (
    <div className="barrier-grid">
      {opts.map(o => (
        <button key={o.id}
                className={`barrier-tile ${value === o.id ? "on" : ""}`}
                onClick={()=>onChange(o.id)}
                title={o.label}>
          <svg viewBox="0 0 48 48">
            <rect x="1" y="1" width="46" height="46" fill="none"
                  stroke="currentColor" strokeOpacity="0.2" strokeDasharray="2 2"/>
            {o.art}
          </svg>
          <span>{o.label}</span>
        </button>
      ))}
    </div>
  );
}

function mulberry32(a: number) {
  return function() {
    let t = a += 0x6D2B79F5;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

export function GridPreview({ size, pop }: { size: number; pop: number }) {
  const density = pop / (size * size);
  const dots = Math.min(60, Math.floor(density * 240));
  const rng = mulberry32(size * 7919 + pop);
  return (
    <div className="grid-preview">
      <svg viewBox="0 0 48 48">
        <rect x="1" y="1" width="46" height="46" fill="var(--bg)"
              stroke="var(--line-2)"/>
        {Array.from({length: dots}, (_, i) => (
          <rect key={i}
                x={2 + rng() * 44} y={2 + rng() * 44}
                width="1.4" height="1.4"
                fill="var(--accent)" opacity={0.6 + rng() * 0.4}/>
        ))}
      </svg>
      <span className="grid-preview-l">{size}²</span>
    </div>
  );
}
