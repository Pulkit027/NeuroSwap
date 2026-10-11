import { ReactNode, useMemo, useState } from "react";
import type { Dataset, Kind } from "./data";
import { fmt } from "./data";

/** Data-source label carried by every chart and table. */
export function SourceLabel({ kind, inputs }: { kind: Kind; inputs?: string[] }) {
  const text = kind === "model_output" ? "model output" : "measured on this machine";
  const detail = kind === "derived_from_measured" ? " (summary of measured runs)" : "";
  return (
    <p className={`source source-${kind === "model_output" ? "model" : "measured"}`}>
      <span className="source-tag">{text}</span>
      {detail}
      {inputs && inputs.length > 0 && <span className="source-inputs"> · {inputs.join(", ")}</span>}
    </p>
  );
}

export function Card({ title, children, kind, inputs, note }: {
  title: string;
  children: ReactNode;
  kind: Kind;
  inputs?: string[];
  note?: ReactNode;
}) {
  return (
    <section className="card">
      <h2>{title}</h2>
      <SourceLabel kind={kind} inputs={inputs} />
      {note && <div className="card-note">{note}</div>}
      {children}
    </section>
  );
}

export interface Column<T> {
  key: string;
  label: string;
  value: (row: T) => string | number | null;
  numeric?: boolean;
  digits?: number;
  render?: (row: T) => ReactNode;
}

/** Sortable table; click a header to sort, again to reverse. */
export function DataTable<T>({ rows, columns, initialSort, maxHeight }: {
  rows: T[];
  columns: Column<T>[];
  initialSort?: string;
  maxHeight?: number;
}) {
  const [sortKey, setSortKey] = useState<string | undefined>(initialSort);
  const [descending, setDescending] = useState(false);
  const sorted = useMemo(() => {
    const column = columns.find((c) => c.key === sortKey);
    if (!column) return rows;
    const copy = [...rows];
    copy.sort((a, b) => {
      const x = column.value(a);
      const y = column.value(b);
      if (x === y) return 0;
      if (x === null) return 1;
      if (y === null) return -1;
      const order = typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y));
      return descending ? -order : order;
    });
    return copy;
  }, [rows, columns, sortKey, descending]);

  const onSort = (key: string) => {
    if (key === sortKey) setDescending(!descending);
    else {
      setSortKey(key);
      setDescending(false);
    }
  };

  return (
    <div className="table-wrap" style={maxHeight ? { maxHeight } : undefined}>
      <table>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key} className={c.numeric ? "num" : undefined}>
                <button type="button" onClick={() => onSort(c.key)} aria-sort={
                  sortKey === c.key ? (descending ? "descending" : "ascending") : "none"}>
                  {c.label}
                  <span className="sort-mark">{sortKey === c.key ? (descending ? " ▾" : " ▴") : ""}</span>
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row, i) => (
            <tr key={i}>
              {columns.map((c) => {
                const v = c.value(row);
                return (
                  <td key={c.key} className={c.numeric ? "num" : undefined}>
                    {c.render ? c.render(row) : typeof v === "number" ? fmt(v, c.digits ?? 3) : v ?? "–"}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Always-visible limitations; numbers come from the JSON where they exist. */
export function Limitations({ data }: { data: Dataset }) {
  const spread = data.policy.data.repetition_spread_s;
  const trainPrompts = data.overview.data.plans.pilot?.prompts;
  return (
    <aside className="limitations" aria-label="Limitations">
      <h2>Limitations</h2>
      <ul>
        <li>ML training data: {trainPrompts ?? "–"} pilot prompts ({data.ml.data.train_rows} runs).</li>
        <li>Pilot "none" runs were warm (no cache drop); final runs, "none" included, were cold.</li>
        <li>{spread.repetitions_per_cell} repetitions per cell in the final batch; no significance test.</li>
        <li>
          Differences between strategies are within repetition noise: the two repetitions of one cell
          differed by a median of {fmt(spread.median, 2)} s (max {fmt(spread.max, 1)} s).
        </li>
        <li>Pilot prefetch used one whole-file madvise (~0.75% resident); it is not a test of prefetch.</li>
        <li>One machine (WSL2), one model, one runtime.</li>
      </ul>
    </aside>
  );
}

export function Select({ label, value, options, onChange }: {
  label: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (value: string) => void;
}) {
  return (
    <label className="filter">
      <span>{label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </label>
  );
}
