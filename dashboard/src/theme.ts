import { useEffect, useState } from "react";

// Chart colors come from CSS custom properties (styles.css), so light and dark use their
// own validated steps. Recharts sets SVG attributes, which cannot read var(), so the
// values are resolved here and re-read whenever the theme changes.

export interface ChartColors {
  series: string[]; // categorical slots in fixed order
  text: string;
  muted: string;
  grid: string;
  surface: string;
}

export type ThemeChoice = "system" | "light" | "dark";

function read(): ChartColors {
  const style = getComputedStyle(document.documentElement);
  const get = (name: string) => style.getPropertyValue(name).trim();
  return {
    series: [1, 2, 3, 4].map((i) => get(`--series-${i}`)),
    text: get("--text-primary"),
    muted: get("--text-secondary"),
    grid: get("--grid"),
    surface: get("--surface-1"),
  };
}

export function useChartColors(): ChartColors {
  const [colors, setColors] = useState<ChartColors>(read);
  useEffect(() => {
    const update = () => setColors(read());
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    media.addEventListener("change", update);
    const observer = new MutationObserver(update);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => {
      media.removeEventListener("change", update);
      observer.disconnect();
    };
  }, []);
  return colors;
}

export function applyTheme(choice: ThemeChoice) {
  if (choice === "system") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = choice;
  try {
    localStorage.setItem("neuroswap-theme", choice);
  } catch {
    /* storage unavailable: the choice just is not remembered */
  }
}

export function storedTheme(): ThemeChoice {
  try {
    const value = localStorage.getItem("neuroswap-theme");
    if (value === "light" || value === "dark" || value === "system") return value;
  } catch {
    /* ignore */
  }
  return "system";
}

/** Fixed color slot per condition, so a condition keeps its color on every chart. */
export const CONDITION_SLOT: Record<string, number> = { baseline: 0, "pin0.25": 1, "pin0.5": 2, prefetch: 3 };
/** Fixed slot per strategy for the 3-series scatter (slots 1-3 validate all-pairs). */
export const STRATEGY_SLOT: Record<string, number> = { baseline: 0, pin: 1, prefetch: 2 };
