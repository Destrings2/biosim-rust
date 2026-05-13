// Persist the single user-editable challenge source in localStorage.

import { BOILERPLATE } from "./template";

const KEY = "biosim:custom_challenge";

export function loadCustomSource(): string {
  try {
    const v = window.localStorage.getItem(KEY);
    return v ?? BOILERPLATE;
  } catch {
    return BOILERPLATE;
  }
}

export function saveCustomSource(source: string): void {
  try {
    window.localStorage.setItem(KEY, source);
  } catch {
    // Quota / privacy-mode — silently drop. The editor still works in-memory.
  }
}
