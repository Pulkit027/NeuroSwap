import { useMemo, useState } from "react";
import { Card, Column, DataTable, Select } from "../components";
import type { Dataset, Run } from "../data";
import { BUCKETS, CONDITION_NAMES, PLANS, conditionOf, fmtInt, sortLimits } from "../data";

const ALL = "__all";

export function Experiments({ data }: { data: Dataset }) {
  const runs = data.experiments.data;
  const [plan, setPlan] = useState(ALL);
  const [bucket, setBucket] = useState(ALL);
  const [limit, setLimit] = useState(ALL);
  const [condition, setCondition] = useState(ALL);
  const [prompt, setPrompt] = useState("");

  const limits = sortLimits([...new Set(runs.map((r) => r.memory_limit))]);
  const conditions = ["baseline", "pin0.25", "pin0.5", "prefetch"].filter((c) =>
    runs.some((r) => conditionOf(r.strategy, r.pin_fraction) === c));

  const filtered = useMemo(() => runs.filter((r) =>
    (plan === ALL || r.plan_name === plan) &&
    (bucket === ALL || r.bucket === bucket) &&
    (limit === ALL || r.memory_limit === limit) &&
    (condition === ALL || conditionOf(r.strategy, r.pin_fraction) === condition) &&
    (prompt === "" || r.prompt_id.includes(prompt.trim()))), [runs, plan, bucket, limit, condition, prompt]);

  const columns: Column<Run>[] = [
    { key: "plan", label: "plan", value: (r) => r.plan_name },
    { key: "prompt", label: "prompt id", value: (r) => r.prompt_id },
    { key: "bucket", label: "bucket", value: (r) => BUCKETS.indexOf(r.bucket), render: (r) => r.bucket },
    { key: "limit", label: "memory limit", value: (r) => r.memory_limit },
    { key: "strategy", label: "strategy", value: (r) => conditionOf(r.strategy, r.pin_fraction),
      render: (r) => CONDITION_NAMES[conditionOf(r.strategy, r.pin_fraction)] ?? r.strategy },
    { key: "rep", label: "rep", value: (r) => r.repetition, numeric: true, render: (r) => String(r.repetition) },
    { key: "cache", label: "cache", value: (r) => (r.cold ? "cold" : "warm") },
    { key: "total", label: "total s", value: (r) => r.total_wall_s, numeric: true },
    { key: "inference", label: "inference s", value: (r) => r.inference_wall_s, numeric: true },
    { key: "setup", label: "setup s", value: (r) => (r.strategy_setup_ms === null ? null : r.strategy_setup_ms / 1000), numeric: true },
    { key: "majflt", label: "major faults", value: (r) => r.ru_majflt, numeric: true, render: (r) => fmtInt(r.ru_majflt) },
    { key: "note", label: "note", value: (r) => r.error ?? r.note ?? "",
      render: (r) => (r.error ? `error: ${r.error}` : r.note ? "single-madvise prefetch" : "") },
  ];

  const options = (values: string[], name = (v: string) => v) =>
    [{ value: ALL, label: "all" }, ...values.map((v) => ({ value: v, label: name(v) }))];

  return (
    <>
      <h1>Experiments</h1>
      <Card title="All runs" kind="measured" inputs={data.experiments.inputs}
        note={<>One row per run (last record per run_id). Setup is the strategy helper's time before
          inference (0 for baseline). Click a column header to sort.</>}>
        <div className="filters">
          <Select label="plan" value={plan} onChange={setPlan} options={options(PLANS.filter((p) => runs.some((r) => r.plan_name === p)))} />
          <Select label="bucket" value={bucket} onChange={setBucket} options={options(BUCKETS)} />
          <Select label="memory limit" value={limit} onChange={setLimit} options={options(limits)} />
          <Select label="strategy" value={condition} onChange={setCondition} options={options(conditions, (c) => CONDITION_NAMES[c] ?? c)} />
          <label className="filter">
            <span>prompt id</span>
            <input value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="e.g. wc0225" />
          </label>
          <span className="filter-count">{filtered.length} of {runs.length} runs</span>
        </div>
        <DataTable rows={filtered} columns={columns} initialSort="plan" maxHeight={640} />
      </Card>
    </>
  );
}
