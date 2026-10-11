import { useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { TooltipProps } from "recharts";
import { Card, Column, DataTable, Select } from "../components";
import type { ComparisonRow, Dataset, PolicyCellRow, PolicySummaryRow } from "../data";
import { BUCKETS, CONDITION_NAMES, PLANS, fmt, fmtInt, sortLimits } from "../data";
import { CONDITION_SLOT, useChartColors } from "../theme";

const CONDITION_ORDER = ["baseline", "pin0.25", "pin0.5", "prefetch"];

function conditionName(condition: string, plan: string): string {
  const name = CONDITION_NAMES[condition] ?? condition;
  return condition === "prefetch" && plan !== "final" ? `${name} (single madvise)` : name;
}

function Legend({ conditions, plan, colors }: { conditions: string[]; plan: string; colors: string[] }) {
  return (
    <ul className="legend">
      {conditions.map((c) => (
        <li key={c}><span className="swatch" style={{ background: colors[CONDITION_SLOT[c]] }} />{conditionName(c, plan)}</li>
      ))}
    </ul>
  );
}

function BarTooltip({ active, payload, label, rows }: TooltipProps<number, string> & { rows: ComparisonRow[] }) {
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div className="tooltip">
      <div className="tooltip-title">bucket {label}</div>
      {payload.map((p) => {
        const row = rows.find((r) => r.bucket === label && r.condition === p.dataKey);
        if (!row) return null;
        return (
          <div key={String(p.dataKey)} className="tooltip-row">
            <span className="swatch" style={{ background: p.color }} />
            <span>{conditionName(row.condition, row.plan_name)}</span>
            <span className="num">median {fmt(row.total_wall_s_median, 2)} s · IQR {fmt(row.total_wall_s_iqr, 2)} ·
              range {fmt(row.total_wall_s_min, 2)}–{fmt(row.total_wall_s_max, 2)} · n {row.n}</span>
          </div>
        );
      })}
    </div>
  );
}

function LimitChart({ rows, limit, plan }: { rows: ComparisonRow[]; limit: string; plan: string }) {
  const colors = useChartColors();
  const conditions = CONDITION_ORDER.filter((c) => rows.some((r) => r.condition === c));
  const chartData = BUCKETS.filter((b) => rows.some((r) => r.bucket === b)).map((bucket) => {
    const entry: Record<string, string | number> = { bucket };
    for (const c of conditions) {
      const row = rows.find((r) => r.bucket === bucket && r.condition === c);
      if (row) entry[c] = row.total_wall_s_median;
    }
    return entry;
  });
  const cache = [...new Set(rows.flatMap((r) => r.cold.map((c) => (c ? "cold" : "warm"))))].join(", ");
  return (
    <figure className="chart">
      <figcaption>Memory limit {limit} · {cache} · median total_wall_s (s) per bucket</figcaption>
      <Legend conditions={conditions} plan={plan} colors={colors.series} />
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={chartData} barGap={2} barCategoryGap="22%" margin={{ top: 8, right: 8, bottom: 4, left: 8 }}>
          <CartesianGrid vertical={false} stroke={colors.grid} />
          <XAxis dataKey="bucket" tick={{ fill: colors.muted, fontSize: 12 }} stroke={colors.grid}
            label={{ value: "prompt tokens bucket", position: "insideBottom", offset: -2, fill: colors.muted, fontSize: 12 }}
            height={40} />
          <YAxis tick={{ fill: colors.muted, fontSize: 12 }} stroke={colors.grid} width={56}
            label={{ value: "seconds", angle: -90, position: "insideLeft", fill: colors.muted, fontSize: 12 }} />
          <Tooltip content={<BarTooltip rows={rows} />} cursor={{ fill: colors.grid, opacity: 0.35 }} />
          {conditions.map((c) => (
            <Bar key={c} dataKey={c} name={conditionName(c, plan)} fill={colors.series[CONDITION_SLOT[c]]}
              radius={[4, 4, 0, 0]} maxBarSize={28} isAnimationActive={false} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </figure>
  );
}

const comparisonColumns: Column<ComparisonRow>[] = [
  { key: "limit", label: "memory limit", value: (r) => r.memory_limit },
  { key: "bucket", label: "bucket", value: (r) => BUCKETS.indexOf(r.bucket), render: (r) => r.bucket },
  { key: "condition", label: "strategy", value: (r) => CONDITION_ORDER.indexOf(r.condition),
    render: (r) => conditionName(r.condition, r.plan_name) },
  { key: "n", label: "runs", value: (r) => r.n, numeric: true, render: (r) => fmtInt(r.n) },
  { key: "prompts", label: "prompts", value: (r) => r.prompts, numeric: true, render: (r) => fmtInt(r.prompts) },
  { key: "median", label: "median total s", value: (r) => r.total_wall_s_median, numeric: true },
  { key: "iqr", label: "IQR s", value: (r) => r.total_wall_s_iqr, numeric: true },
  { key: "min", label: "min s", value: (r) => r.total_wall_s_min, numeric: true },
  { key: "max", label: "max s", value: (r) => r.total_wall_s_max, numeric: true },
  { key: "inference", label: "median inference s", value: (r) => r.inference_wall_s_median, numeric: true },
  { key: "setup", label: "median setup ms", value: (r) => r.strategy_setup_ms_median, numeric: true, digits: 1 },
  { key: "majflt", label: "median major faults", value: (r) => r.ru_majflt_median, numeric: true, render: (r) => fmtInt(r.ru_majflt_median) },
];

function policyColumns(rows: PolicySummaryRow[]): Column<PolicySummaryRow>[] {
  const keys = Object.keys(rows[0] ?? {});
  const means = keys.filter((k) => k.startsWith("mean_s_"));
  const picks = keys.filter((k) => k.startsWith("picked_"));
  const num = (k: string) => (r: PolicySummaryRow) => (typeof r[k] === "number" ? (r[k] as number) : null);
  return [
    { key: "policy", label: "policy", value: (r) => String(r.policy) },
    { key: "cells", label: "cells", value: num("cells"), numeric: true, render: (r) => fmtInt(r.cells as number) },
    { key: "mean_s", label: "mean s", value: num("mean_s"), numeric: true },
    ...means.map((k) => ({ key: k, label: `mean s, ${k.slice(7)}`, value: num(k), numeric: true })),
    ...picks.map((k) => ({ key: k, label: `picked ${CONDITION_NAMES[k.slice(7)] ?? k.slice(7)}`, value: num(k),
      numeric: true, render: (r: PolicySummaryRow) => fmtInt(r[k] as number) })),
    { key: "mean_diff", label: "mean diff vs baseline s", value: num("mean_diff_vs_baseline_s"), numeric: true },
    { key: "median_diff", label: "median diff vs baseline s", value: num("median_diff_vs_baseline_s"), numeric: true },
    { key: "lower", label: "cells lower", value: num("cells_lower"), numeric: true, render: (r) => fmtInt(r.cells_lower as number) },
    { key: "higher", label: "cells higher", value: num("cells_higher"), numeric: true, render: (r) => fmtInt(r.cells_higher as number) },
    { key: "equal", label: "cells equal", value: num("cells_equal"), numeric: true, render: (r) => fmtInt(r.cells_equal as number) },
  ];
}

function cellColumns(rows: PolicyCellRow[]): Column<PolicyCellRow>[] {
  const policies = Object.keys(rows[0] ?? {}).filter((k) => k.endsWith(": diff_s")).map((k) => k.slice(0, -": diff_s".length));
  return [
    { key: "prompt", label: "prompt id", value: (r) => String(r.prompt_id) },
    { key: "limit", label: "memory limit", value: (r) => String(r.memory_limit) },
    { key: "bucket", label: "bucket", value: (r) => BUCKETS.indexOf(String(r.bucket)), render: (r) => String(r.bucket) },
    { key: "tokens", label: "prompt tokens", value: (r) => r.prompt_tokens as number, numeric: true, render: (r) => fmtInt(r.prompt_tokens as number) },
    { key: "baseline", label: "always-baseline s", value: (r) => r.baseline_s as number, numeric: true },
    ...policies.map((p) => ({
      key: p, label: `${p} − baseline s`, value: (r: PolicyCellRow) => r[`${p}: diff_s`] as number, numeric: true,
      render: (r: PolicyCellRow) => (
        <>
          {fmt(r[`${p}: diff_s`] as number, 3)}
          <span className="chosen"> {CONDITION_NAMES[String(r[`${p}: chosen`])] ?? String(r[`${p}: chosen`])}</span>
        </>
      ),
    })),
  ];
}

export function StrategyComparison({ data }: { data: Dataset }) {
  const available = PLANS.filter((p) => data.comparison.data.some((r) => r.plan_name === p));
  const [plan, setPlan] = useState(available.includes("final") ? "final" : available[0]);
  const rows = data.comparison.data.filter((r) => r.plan_name === plan);
  const limits = sortLimits([...new Set(rows.map((r) => r.memory_limit))]);
  const policy = data.policy.data;

  return (
    <>
      <h1>Strategy Comparison</h1>
      <Card title="Median total latency per bucket and memory limit" kind="derived_from_measured"
        inputs={data.comparison.inputs}
        note={<>
          total_wall_s includes the strategy's setup time. Pin 50% exists only in the pilot.
          {plan !== "final" && <> Pilot prefetch used one whole-file madvise (~0.75% resident) and is not a test of prefetch.</>}
          {plan === "pilot" && <> Pilot "none" runs were warm.</>}
        </>}>
        <div className="filters">
          <Select label="plan" value={plan} onChange={setPlan} options={available.map((p) => ({ value: p, label: p }))} />
        </div>
        <div className="chart-grid">
          {limits.map((limit) => (
            <LimitChart key={`${plan}-${limit}`} rows={rows.filter((r) => r.memory_limit === limit)} limit={limit} plan={plan} />
          ))}
        </div>
        <details className="table-toggle">
          <summary>Table view ({rows.length} rows)</summary>
          <DataTable rows={rows} columns={comparisonColumns} initialSort="limit" />
        </details>
      </Card>

      <Card title="Policy results on the held-out final batch" kind="derived_from_measured" inputs={data.policy.inputs}
        note={<>
          {policy.latency}. Rule threshold chosen on train: prompt_tokens &gt; {policy.rule_threshold_prompt_tokens}.
          Two repetitions per cell; median spread between them {fmt(policy.repetition_spread_s.median, 2)} s.
        </>}>
        <DataTable rows={policy.summary} columns={policyColumns(policy.summary)} />
        <p className="statement">{policy.statement}</p>
      </Card>

      <Card title="Paired differences vs always-baseline, per prompt and memory limit" kind="derived_from_measured"
        inputs={data.policy.inputs}
        note={<>Policy latency minus always-baseline latency for the same prompt and limit (negative = lower
          measured latency). The condition each policy chose is shown next to the value.</>}>
        <DataTable rows={policy.per_cell} columns={cellColumns(policy.per_cell)} initialSort="limit" maxHeight={560} />
      </Card>
    </>
  );
}
