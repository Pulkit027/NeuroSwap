import { Card, Column, DataTable } from "../components";
import type { Dataset, Run } from "../data";
import { PLANS, fmt, fmtInt, median, sortLimits } from "../data";

interface LimitRow {
  plan: string;
  limit: string;
  cold: string;
  n: number;
  total: number | null;
  inference: number | null;
  tps: number | null;
  majflt: number | null;
  rssMiB: number | null;
}

function byLimit(runs: Run[]): LimitRow[] {
  const rows: LimitRow[] = [];
  for (const plan of PLANS) {
    const planRuns = runs.filter((r) => r.plan_name === plan && r.error === null);
    for (const limit of sortLimits([...new Set(planRuns.map((r) => r.memory_limit))])) {
      const g = planRuns.filter((r) => r.memory_limit === limit);
      const nums = (pick: (r: Run) => number | null) =>
        g.map(pick).filter((v): v is number => v !== null);
      rows.push({
        plan,
        limit,
        cold: [...new Set(g.map((r) => (r.cold ? "cold" : "warm")))].join(", "),
        n: g.length,
        total: median(nums((r) => r.total_wall_s)),
        inference: median(nums((r) => r.inference_wall_s)),
        tps: median(nums((r) => r.eval_tps)),
        majflt: median(nums((r) => r.ru_majflt)),
        rssMiB: median(nums((r) => (r.ru_maxrss_kb === null ? null : r.ru_maxrss_kb / 1024))),
      });
    }
  }
  return rows;
}

const limitColumns: Column<LimitRow>[] = [
  { key: "plan", label: "plan", value: (r) => r.plan },
  { key: "limit", label: "memory limit", value: (r) => r.limit },
  { key: "cold", label: "cache", value: (r) => r.cold },
  { key: "n", label: "runs", value: (r) => r.n, numeric: true, render: (r) => fmtInt(r.n) },
  { key: "total", label: "median total s", value: (r) => r.total, numeric: true },
  { key: "inference", label: "median inference s", value: (r) => r.inference, numeric: true },
  { key: "tps", label: "median decode tokens/s", value: (r) => r.tps, numeric: true, digits: 2 },
  { key: "majflt", label: "median major faults", value: (r) => r.majflt, numeric: true, render: (r) => fmtInt(r.majflt) },
  { key: "rss", label: "median peak RSS MiB", value: (r) => r.rssMiB, numeric: true, digits: 1 },
];

export function Overview({ data }: { data: Dataset }) {
  const plans = data.overview.data.plans;
  const totalRuns = Object.values(plans).reduce((s, p) => s + p.runs, 0);
  const totalErrors = Object.values(plans).reduce((s, p) => s + p.errors, 0);
  const commits = [...new Set(Object.values(plans).flatMap((p) => p.llama_cpp_commit))];
  const shas = [...new Set(Object.values(plans).flatMap((p) => p.model_sha256))];
  const repack = [...new Set(Object.values(plans).flatMap((p) => p.repack))];
  const rawInputs = data.experiments.inputs;

  return (
    <>
      <h1>Overview</h1>

      <section className="card measured-box">
        <h2>What was measured</h2>
        <dl>
          <dt>Model</dt>
          <dd>Qwen2.5-1.5B-Instruct, GGUF Q4_K_M (sha256 {shas.map((s) => s.slice(0, 12)).join(", ")}…)</dd>
          <dt>Runtime</dt>
          <dd>llama.cpp llama-completion (commit {commits.map((c) => c.slice(0, 9)).join(", ")}), --no-repack
            (repack: {repack.join(", ")}), 64 new tokens, context 2048, 4 threads, temperature 0</dd>
          <dt>Dataset</dt>
          <dd>WildChat-1M user prompts (ODC-BY), first user turn, 300-prompt manifest in four token-length buckets</dd>
          <dt>Strategies</dt>
          <dd>baseline, prefetch (madvise WILLNEED on the model file), pin (mlock of a fraction of the
            weights) under cgroup memory limits; the KV cache is not controlled</dd>
        </dl>
      </section>

      <Card title="Runs by plan" kind="measured" inputs={rawInputs}>
        <div className="tiles">
          <div className="tile"><span className="tile-value">{fmtInt(totalRuns)}</span><span className="tile-label">runs</span></div>
          <div className="tile"><span className="tile-value">{fmtInt(totalErrors)}</span><span className="tile-label">runs with an error</span></div>
        </div>
        <DataTable
          rows={PLANS.filter((p) => plans[p]).map((p) => ({ plan: p, ...plans[p] }))}
          columns={[
            { key: "plan", label: "plan", value: (r) => r.plan },
            { key: "runs", label: "runs", value: (r) => r.runs, numeric: true, render: (r) => fmtInt(r.runs) },
            { key: "errors", label: "errors", value: (r) => r.errors, numeric: true, render: (r) => fmtInt(r.errors) },
            { key: "prompts", label: "prompts", value: (r) => r.prompts, numeric: true, render: (r) => fmtInt(r.prompts) },
            { key: "limits", label: "memory limits", value: (r) => r.memory_limits.join(", ") },
            { key: "conditions", label: "conditions", value: (r) => r.conditions.join(", ") },
            { key: "cold", label: "cache", value: (r) => r.cold.map((c) => (c ? "cold" : "warm")).join(", ") },
            { key: "when", label: "last session (UTC)", value: (r) =>
              r.session_started_at ? `${r.session_started_at.slice(0, 16)} – ${(r.session_updated_at ?? "").slice(11, 16)}` : "–" },
          ]}
        />
      </Card>

      <Card title="Medians by plan and memory limit" kind="derived_from_measured" inputs={rawInputs}
        note={<>Error-free runs, all strategies pooled. Pilot "none" runs were warm; final runs were cold, so
          rows of different plans are not directly comparable.</>}>
        <DataTable rows={byLimit(data.experiments.data)} columns={limitColumns} />
      </Card>

      <Card title="Policy evaluation statement" kind="derived_from_measured" inputs={data.policy.inputs}>
        <p>{data.overview.data.policy_statement}</p>
        <p className="muted">
          Repetition spread on the test set: median {fmt(data.policy.data.repetition_spread_s.median, 2)} s,
          max {fmt(data.policy.data.repetition_spread_s.max, 1)} s between the{" "}
          {data.policy.data.repetition_spread_s.repetitions_per_cell} repetitions of one cell.
        </p>
      </Card>
    </>
  );
}
