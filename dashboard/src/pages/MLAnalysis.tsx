import {
  Bar, BarChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";
import type { TooltipProps } from "recharts";
import { Card, Column, DataTable } from "../components";
import type { Dataset, Metric, PredictionRow } from "../data";
import { fmt, fmtInt } from "../data";
import { STRATEGY_SLOT, useChartColors } from "../theme";

const MODEL_NAMES: Record<string, string> = {
  median_bucket_limit: "(a) median per bucket + limit",
  linear_tokens_per_limit: "(b) linear on prompt_tokens, per limit",
  random_forest: "(c) random forest (seed 42)",
};

interface MetricRow {
  model: string;
  selected: boolean;
  evaluation: string;
  rows: string;
  metric: Metric;
}

function metricRows(data: Dataset): MetricRow[] {
  const ml = data.ml.data;
  const out: MetricRow[] = [];
  for (const [model, metric] of Object.entries(ml.cv_train)) {
    out.push({ model, selected: model === ml.selected_model, evaluation: "CV on train (leave one prompt out)",
      rows: "all train rows", metric });
  }
  for (const [model, subsets] of Object.entries(ml.test)) {
    for (const [subset, metric] of Object.entries(subsets)) {
      out.push({ model, selected: model === ml.selected_model, evaluation: "held-out test (final batch)", rows: subset, metric });
    }
  }
  return out;
}

const metricColumns: Column<MetricRow>[] = [
  { key: "model", label: "model", value: (r) => r.model,
    render: (r) => <>{MODEL_NAMES[r.model] ?? r.model}{r.selected && <span className="badge">selected by CV</span>}</> },
  { key: "evaluation", label: "evaluation", value: (r) => r.evaluation },
  { key: "rows", label: "rows", value: (r) => r.rows,
    render: (r) => (r.rows.includes("unseen") ? <strong>{r.rows}</strong> : r.rows) },
  { key: "n", label: "n", value: (r) => r.metric.n, numeric: true, render: (r) => fmtInt(r.metric.n) },
  { key: "mae", label: "MAE s", value: (r) => r.metric.mae, numeric: true },
  { key: "rmse", label: "RMSE s", value: (r) => r.metric.rmse, numeric: true },
  { key: "r2", label: "R²", value: (r) => r.metric.r2, numeric: true },
];

function ScatterTooltip({ active, payload }: TooltipProps<number, string>) {
  if (!active || !payload || payload.length === 0) return null;
  const row = payload[0].payload as PredictionRow;
  return (
    <div className="tooltip">
      <div className="tooltip-title">{row.prompt_id} · {row.bucket} · {row.memory_limit} · {row.strategy}
        {row.strategy === "pin" ? ` ${row.pin_fraction}` : ""}</div>
      <div className="tooltip-row"><span>measured</span><span className="num">{fmt(row.actual, 3)} s</span></div>
      <div className="tooltip-row"><span>predicted</span><span className="num">{fmt(row.predicted, 3)} s</span></div>
    </div>
  );
}

function PredictionScatter({ data }: { data: Dataset }) {
  const colors = useChartColors();
  const rows = data.predictions.data.rows;
  const strategies = ["baseline", "pin", "prefetch"].filter((s) => rows.some((r) => r.strategy === s));
  const max = Math.ceil(Math.max(...rows.map((r) => Math.max(r.actual, r.predicted))) / 10) * 10;
  return (
    <figure className="chart">
      <figcaption>x: measured total_wall_s (s) · y: predicted (s) · dashed line: predicted = measured</figcaption>
      <ul className="legend">
        {strategies.map((s) => (
          <li key={s}><span className="swatch round" style={{ background: colors.series[STRATEGY_SLOT[s]] }} />
            {s === "prefetch" ? "prefetch (unseen in training)" : s === "pin" ? "pin 25%" : s}</li>
        ))}
      </ul>
      <ResponsiveContainer width="100%" height={380}>
        <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid stroke={colors.grid} />
          <XAxis type="number" dataKey="actual" domain={[0, max]} tick={{ fill: colors.muted, fontSize: 12 }} stroke={colors.grid}
            label={{ value: "measured s", position: "insideBottom", offset: -4, fill: colors.muted, fontSize: 12 }} height={44} />
          <YAxis type="number" dataKey="predicted" domain={[0, max]} tick={{ fill: colors.muted, fontSize: 12 }} stroke={colors.grid}
            label={{ value: "predicted s", angle: -90, position: "insideLeft", fill: colors.muted, fontSize: 12 }} width={56} />
          <ZAxis range={[64, 64]} />
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: max, y: max }]} stroke={colors.muted} strokeDasharray="4 4" ifOverflow="extendDomain" />
          <Tooltip content={<ScatterTooltip />} cursor={{ stroke: colors.grid }} />
          {strategies.map((s) => (
            <Scatter key={s} name={s} data={rows.filter((r) => r.strategy === s)} fill={colors.series[STRATEGY_SLOT[s]]}
              stroke={colors.surface} strokeWidth={2} isAnimationActive={false} />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </figure>
  );
}

function Importances({ data }: { data: Dataset }) {
  const colors = useChartColors();
  const rows = Object.entries(data.importance.data.importances)
    .map(([feature, importance]) => ({ feature, importance }))
    .sort((a, b) => b.importance - a.importance);
  return (
    <>
      <ResponsiveContainer width="100%" height={40 + rows.length * 34}>
        <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 48, bottom: 4, left: 8 }}>
          <CartesianGrid horizontal={false} stroke={colors.grid} />
          <XAxis type="number" domain={[0, 1]} tick={{ fill: colors.muted, fontSize: 12 }} stroke={colors.grid} />
          <YAxis type="category" dataKey="feature" width={140} tick={{ fill: colors.text, fontSize: 12 }} stroke={colors.grid} />
          <Tooltip cursor={{ fill: colors.grid, opacity: 0.35 }}
            content={({ active, payload }) => active && payload?.length ? (
              <div className="tooltip"><div className="tooltip-row">
                <span>{String(payload[0].payload.feature)}</span><span className="num">{fmt(Number(payload[0].value), 4)}</span>
              </div></div>) : null} />
          <Bar dataKey="importance" fill={colors.series[0]} radius={[0, 4, 4, 0]} maxBarSize={22} isAnimationActive={false}
            label={{ position: "right", fill: colors.text, fontSize: 12, formatter: (v: number) => fmt(v, 3) }} />
        </BarChart>
      </ResponsiveContainer>
      <details className="table-toggle">
        <summary>Table view</summary>
        <DataTable rows={rows} columns={[
          { key: "feature", label: "feature", value: (r) => r.feature },
          { key: "importance", label: "importance", value: (r) => r.importance, numeric: true, digits: 4 },
        ]} />
      </details>
    </>
  );
}

export function MLAnalysis({ data }: { data: Dataset }) {
  const ml = data.ml.data;
  const predictions = data.predictions.data;
  return (
    <>
      <h1>ML Analysis</h1>
      <Card title="Model comparison (target total_wall_s)" kind="derived_from_measured" inputs={data.ml.inputs}
        note={<>
          Selection: {ml.selection}. Train: {ml.train_rows} pilot runs; test: {ml.test_rows} final-batch runs (held out).
          Prefetch does not occur in train, so the prefetch rows are an unseen strategy for every model and are
          reported separately. {ml.notes.filter((n) => !n.includes("unseen")).join(" ")}
        </>}>
        <DataTable rows={metricRows(data)} columns={metricColumns} />
      </Card>

      <Card title={`Predicted vs measured, held-out test (${MODEL_NAMES[predictions.model] ?? predictions.model})`}
        kind="model_output" inputs={data.predictions.inputs}
        note={<>Predictions are model output; the measured axis is {predictions.actual}. The selected model ignores
          strategy, so all strategies of one prompt and limit get the same prediction.</>}>
        <PredictionScatter data={data} />
        <details className="table-toggle">
          <summary>Table view ({predictions.rows.length} rows)</summary>
          <DataTable rows={predictions.rows} initialSort="prompt" maxHeight={420} columns={[
            { key: "prompt", label: "prompt id", value: (r) => r.prompt_id },
            { key: "limit", label: "memory limit", value: (r) => r.memory_limit },
            { key: "strategy", label: "strategy", value: (r) => (r.strategy === "pin" ? `pin ${r.pin_fraction}` : r.strategy) },
            { key: "actual", label: "measured s", value: (r) => r.actual, numeric: true },
            { key: "predicted", label: "predicted s", value: (r) => r.predicted, numeric: true },
            { key: "error", label: "predicted − measured s", value: (r) => r.predicted - r.actual, numeric: true },
          ]} />
        </details>
      </Card>

      <Card title="Random forest feature importances" kind="model_output" inputs={data.importance.inputs}
        note={<>{data.importance.data.method}. strategy_prefetch is 0 by construction (prefetch never occurs in train).</>}>
        <Importances data={data} />
      </Card>
    </>
  );
}
