// Minimal line icons used throughout the app. 1.5px stroke, currentColor.

import type { ReactNode } from "react";

interface IconProps {
  size?: number;
  sw?: number;
  fill?: string;
  d?: string;
  children?: ReactNode;
}

const Icon = ({ d, size = 16, sw = 1.5, fill = "none", children }: IconProps) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill={fill}
       stroke="currentColor" strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round">
    {d ? <path d={d} /> : children}
  </svg>
);

type P = Omit<IconProps, "d" | "children">;

export const IcInspect = (p: P) => (
  <Icon {...p}><circle cx="11" cy="11" r="6" /><path d="M16 16 L21 21" /></Icon>
);
export const IcBarrier = (p: P) => (
  <Icon {...p}>
    <rect x="3" y="5" width="18" height="14" rx="1" />
    <path d="M3 10 H21 M3 15 H21 M9 5 V10 M14 10 V15 M9 15 V19 M16 5 V10" />
  </Icon>
);
export const IcKill = (p: P) => <Icon {...p}><path d="M5 5 L19 19 M19 5 L5 19" /></Icon>;
export const IcReproduce = (p: P) => (
  <Icon {...p}><path d="M12 4 V20 M4 12 H20" /><circle cx="12" cy="12" r="3" /></Icon>
);
export const IcPlay = (p: P) => <Icon {...p} fill="currentColor"><path d="M7 5 L19 12 L7 19 Z" /></Icon>;
export const IcPause = (p: P) => (
  <Icon {...p} fill="currentColor"><rect x="6" y="5" width="4" height="14" /><rect x="14" y="5" width="4" height="14" /></Icon>
);
export const IcStep = (p: P) => <Icon {...p}><path d="M5 5 V19 M9 12 L19 12 M14 7 L19 12 L14 17" /></Icon>;
export const IcStepGen = (p: P) => <Icon {...p}><path d="M5 5 V19 M19 5 V19 M9 12 H15 M12 9 L15 12 L12 15" /></Icon>;
export const IcEpoch = (p: P) => (
  <Icon {...p}><path d="M21 12 a9 9 0 1 1 -3.5 -7.1" /><path d="M21 4 V8 H17" /></Icon>
);
export const IcReset = (p: P) => (
  <Icon {...p}><path d="M3 12 a9 9 0 1 0 3-6.7" /><path d="M3 4 V9 H8" /></Icon>
);
export const IcStats = (p: P) => (
  <Icon {...p}><path d="M4 19 H20" /><path d="M7 16 V11 M12 16 V6 M17 16 V13" /></Icon>
);
export const IcChallenge = (p: P) => (
  <Icon {...p}><circle cx="12" cy="12" r="8" /><circle cx="12" cy="12" r="3" /><path d="M12 4 V8 M12 16 V20 M4 12 H8 M16 12 H20" /></Icon>
);
export const IcRegistry = (p: P) => <Icon {...p}><path d="M4 6 H20 M4 12 H20 M4 18 H14" /></Icon>;
export const IcConfig = (p: P) => (
  <Icon {...p}><path d="M5 7 L9 11 L5 15" /><path d="M12 17 H19" /></Icon>
);
export const IcTelemetry = (p: P) => (
  <Icon {...p}>
    <path d="M3 16 L8 11 L12 14 L17 7 L21 11" />
    <circle cx="8" cy="11" r="1" fill="currentColor"/>
    <circle cx="17" cy="7" r="1" fill="currentColor"/>
  </Icon>
);
export const IcClose = (p: P) => <Icon {...p} d="M6 6 L18 18 M18 6 L6 18" />;
export const IcSearch = (p: P) => <Icon {...p}><circle cx="11" cy="11" r="6" /><path d="M16 16 L21 21" /></Icon>;
export const IcKeyboard = (p: P) => (
  <Icon {...p}>
    <rect x="3" y="7" width="18" height="11" rx="1.5"/>
    <path d="M7 11 H7.01 M11 11 H11.01 M15 11 H15.01 M7 14 H17"/>
  </Icon>
);
