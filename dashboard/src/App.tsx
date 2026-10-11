import { useEffect, useState } from "react";
import { Limitations } from "./components";
import type { Dataset } from "./data";
import { loadDataset } from "./data";
import { Experiments } from "./pages/Experiments";
import { MLAnalysis } from "./pages/MLAnalysis";
import { Overview } from "./pages/Overview";
import { StrategyComparison } from "./pages/StrategyComparison";
import { ThemeChoice, applyTheme, storedTheme } from "./theme";

const PAGES = [
  { id: "overview", label: "Overview", render: (d: Dataset) => <Overview data={d} /> },
  { id: "experiments", label: "Experiments", render: (d: Dataset) => <Experiments data={d} /> },
  { id: "strategies", label: "Strategy Comparison", render: (d: Dataset) => <StrategyComparison data={d} /> },
  { id: "ml", label: "ML Analysis", render: (d: Dataset) => <MLAnalysis data={d} /> },
];

function currentPage(): string {
  const id = window.location.hash.replace(/^#\/?/, "");
  return PAGES.some((p) => p.id === id) ? id : "overview";
}

export function App() {
  const [data, setData] = useState<Dataset | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(currentPage);
  const [theme, setTheme] = useState<ThemeChoice>(storedTheme);

  useEffect(() => {
    loadDataset().then(setData).catch((e: unknown) => setError(String(e)));
    const onHash = () => setPage(currentPage());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => applyTheme(theme), [theme]);

  const active = PAGES.find((p) => p.id === page) ?? PAGES[0];
  return (
    <div className="layout">
      <nav className="sidebar" aria-label="Pages">
        <div className="brand">NeuroSwap</div>
        <ul>
          {PAGES.map((p) => (
            <li key={p.id}>
              <a href={`#/${p.id}`} aria-current={p.id === page ? "page" : undefined}>{p.label}</a>
            </li>
          ))}
        </ul>
        <label className="theme">
          <span>theme</span>
          <select value={theme} onChange={(e) => setTheme(e.target.value as ThemeChoice)}>
            <option value="system">system</option>
            <option value="light">light</option>
            <option value="dark">dark</option>
          </select>
        </label>
        {data && <Limitations data={data} />}
      </nav>
      <main>
        {error && <p className="error">Could not load dashboard data: {error}</p>}
        {!data && !error && <p className="muted">Loading data…</p>}
        {data && active.render(data)}
        {data && (
          <footer className="muted">
            Data exported {data.overview.generated_at} · source field: {data.overview.source}
          </footer>
        )}
      </main>
    </div>
  );
}
