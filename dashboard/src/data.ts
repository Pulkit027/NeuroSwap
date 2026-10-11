// Types and loading for public/data/*.json (written by scripts/export_dashboard_json.py,
// copied by scripts/sync_dashboard_data.sh). Field names match the JSON exactly.

export type Kind = "measured" | "derived_from_measured" | "model_output";

export interface Envelope<T> {
  source: string;
  kind: Kind;
  generated_at: string;
  inputs: string[];
  data: T;
}

export interface PlanInfo {
  runs: number;
  errors: number;
  prompts: number;
  memory_limits: string[];
  conditions: string[];
  cold: boolean[];
  session_started_at: string | null;
  session_updated_at: string | null;
  llama_cpp_commit: string[];
  model_sha256: string[];
  repack: string[];
}

export interface Overview {
  plans: Record<string, PlanInfo>;
  ml: { train_rows: number; test_rows: number; selected_model: string; selection: string };
  policy_statement: string;
  notes: string[];
}

export interface Run {
  run_id: string;
  plan_name: string;
  prompt_id: string;
  bucket: string;
  prompt_tokens: number;
  memory_limit: string;
  strategy: string;
  pin_fraction: number;
  repetition: number;
  cold: boolean;
  error: string | null;
  total_wall_s: number | null;
  inference_wall_s: number | null;
  strategy_setup_ms: number | null;
  ru_majflt: number | null;
  ru_maxrss_kb: number | null;
  eval_tps: number | null;
  prefetch_final_percent: number | null;
  note: string | null;
}

export interface ComparisonRow {
  plan_name: string;
  bucket: string;
  memory_limit: string;
  condition: string;
  n: number;
  prompts: number;
  cold: boolean[];
  total_wall_s_median: number;
  total_wall_s_iqr: number;
  total_wall_s_min: number;
  total_wall_s_max: number;
  inference_wall_s_median: number;
  strategy_setup_ms_median: number;
  ru_majflt_median: number;
  note: string | null;
}

export interface Metric {
  n: number;
  mae: number | null;
  rmse: number | null;
  r2: number | null;
}

export interface MlMetrics {
  source: string;
  target: string;
  selected_model: string;
  selection: string;
  train_rows: number;
  test_rows: number;
  cv_train: Record<string, Metric>;
  test: Record<string, Record<string, Metric>>;
  notes: string[];
}

export interface FeatureImportance {
  source: string;
  model: string;
  method: string;
  importances: Record<string, number>;
}

export interface PredictionRow {
  run_id: string;
  actual: number;
  predicted: number;
  prompt_id: string;
  bucket: string;
  prompt_tokens: number;
  memory_limit: string;
  strategy: string;
  pin_fraction: number;
}

export interface Predictions {
  model: string;
  actual: string;
  predicted: string;
  rows: PredictionRow[];
}

export type PolicySummaryRow = Record<string, string | number>;
export type PolicyCellRow = Record<string, string | number>;

export interface PolicyResults {
  source: string;
  scored_on: string;
  latency: string;
  rule_threshold_prompt_tokens: number;
  summary: PolicySummaryRow[];
  per_cell: PolicyCellRow[];
  statement: string;
  repetition_spread_s: { median: number; max: number; repetitions_per_cell: number };
}

export interface Dataset {
  overview: Envelope<Overview>;
  experiments: Envelope<Run[]>;
  comparison: Envelope<ComparisonRow[]>;
  ml: Envelope<MlMetrics>;
  importance: Envelope<FeatureImportance>;
  predictions: Envelope<Predictions>;
  policy: Envelope<PolicyResults>;
}

async function load<T>(name: string): Promise<Envelope<T>> {
  const response = await fetch(`${import.meta.env.BASE_URL}data/${name}.json`);
  if (!response.ok) throw new Error(`data/${name}.json: HTTP ${response.status}`);
  return (await response.json()) as Envelope<T>;
}

export async function loadDataset(): Promise<Dataset> {
  const [overview, experiments, comparison, ml, importance, predictions, policy] = await Promise.all([
    load<Overview>("overview"),
    load<Run[]>("experiments"),
    load<ComparisonRow[]>("strategy_comparison"),
    load<MlMetrics>("ml_metrics"),
    load<FeatureImportance>("feature_importance"),
    load<Predictions>("predictions"),
    load<PolicyResults>("policy_results"),
  ]);
  return { overview, experiments, comparison, ml, importance, predictions, policy };
}

// ---- helpers ----

export const BUCKETS = ["<64", "64-256", "256-1024", "1024-1500"];
export const PLANS = ["pilot", "pilot_tight", "final"];

/** Condition label used in the JSON: "baseline", "prefetch", "pin0.25", "pin0.5". */
export function conditionOf(strategy: string, pinFraction: number): string {
  return strategy === "pin" ? `pin${pinFraction}` : strategy;
}

export const CONDITION_NAMES: Record<string, string> = {
  baseline: "baseline",
  "pin0.25": "pin 25%",
  "pin0.5": "pin 50%",
  prefetch: "prefetch",
};

export function limitMb(limit: string): number {
  if (limit === "none") return Number.POSITIVE_INFINITY;
  const units: Record<string, number> = { K: 1 / 1024, M: 1, G: 1024 };
  const unit = limit.slice(-1);
  return unit in units ? Number(limit.slice(0, -1)) * units[unit] : Number(limit) / 1048576;
}

/** Limits ordered none first, then from largest to smallest. */
export function sortLimits(limits: string[]): string[] {
  return [...limits].sort((a, b) => limitMb(b) - limitMb(a));
}

export function median(values: number[]): number | null {
  const v = values.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
  if (v.length === 0) return null;
  const mid = Math.floor(v.length / 2);
  return v.length % 2 ? v[mid] : (v[mid - 1] + v[mid]) / 2;
}

export function fmt(value: number | null | undefined, digits = 3): string {
  return value === null || value === undefined || !Number.isFinite(value) ? "–" : value.toFixed(digits);
}

export function fmtInt(value: number | null | undefined): string {
  return value === null || value === undefined || !Number.isFinite(value) ? "–" : Math.round(value).toLocaleString("en-US");
}
