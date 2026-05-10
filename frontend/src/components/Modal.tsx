import { useEffect, useRef } from "react";
import { createPortal } from "react-dom";

/** Renders children into document.body via a portal, bypassing any stacking
 *  context created by backdrop-filter / transforms in ancestor elements. */
export function Modal({ children }: { children: React.ReactNode }) {
  const el = useRef(document.createElement("div"));

  useEffect(() => {
    const node = el.current;
    document.body.appendChild(node);
    return () => { document.body.removeChild(node); };
  }, []);

  return createPortal(children, el.current);
}
