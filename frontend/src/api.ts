const BASE = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") ?? "";

export type ModelKey = "pca_t2_q" | "lstm_autoencoder" | "local_outlier_factor" | "isolation_forest";

export const MODEL_LABELS: Record<ModelKey, string> = {
  pca_t2_q: "PCA T² + Q",
  lstm_autoencoder: "LSTM Autoencoder",
  local_outlier_factor: "Local Outlier Factor",
  isolation_forest: "Isolation Forest",
};

export const MODEL_COLORS: Record<ModelKey, string> = {
  pca_t2_q: "#f59e0b",
  lstm_autoencoder: "#38bdf8",
  local_outlier_factor: "#a78bfa",
  isolation_forest: "#94a3b8",
};

export interface Contributor {
  feature: string;
  description: string;
  value: number;
  share_pct: number;
}

export interface FaultEvent {
  event_id: number;
  start: string;
  end: string;
  start_idx: number;
  end_idx: number;
  duration_min: number;
  peak_score: number;
  mean_score: number;
  contributors: Contributor[];
}

export interface ModelResult {
  model: ModelKey;
  model_name: string;
  contribution_method: string;
  threshold: number;
  summary: {
    rows: number;
    start: string;
    end: string;
    raw_alarm_pct: number;
    event_pct: number;
    minutes_in_fault: number;
    events: number;
    mean_score: number;
    max_score: number;
  };
  overall_contributors: Contributor[];
  events: FaultEvent[];
  series: { t: string; score: number; event: boolean }[];
  series_bucket_min: number;
}

export interface AnalysisResponse {
  results: ModelResult[];
  warnings: string[];
}

export interface LeaderboardRow {
  rank: number;
  is_best: boolean;
  model_key: ModelKey;
  display_name: string;
  family: string;
  best_epoch_or_stage: string;
  composite_score: number;
  train_rmse: number;
  val_rmse: number;
  test_rmse: number;
  test_roc_auc: number;
  test_f1: number;
  val_f1: number;
  threshold_99pct: number;
  test_precision: number;
  test_recall: number;
  test_accuracy: number;
  test_events: number;
  test_snr: number;
}

export interface GDRow {
  optimizer_key: string;
  optimizer_name: string;
  batch_size: string;
  updates_per_epoch: number;
  train_cost_function_mse: number;
  train_rmse: number;
  val_rmse: number;
  test_rmse: number;
  test_f1_score: number;
  test_roc_auc: number;
  test_accuracy: number;
  composite_score: number;
  train_time_s: number;
}

export interface Overview {
  best_model: ModelKey;
  leaderboard: LeaderboardRow[];
  splits: {
    total_rows: number;
    columns: number;
    splits: Record<string, { rows: number; ratio: number; start_timestamp: string; end_timestamp: string }>;
  };
  gradient_descent: { gradient_descent_comparison: GDRow[] };
  features: string[];
  test_window: { start: string; end: string; rows: number; hours: number };
}

export interface ModelInfo {
  key: ModelKey;
  name: string;
  family: string;
  threshold: number;
  contribution_method: string;
  config: Record<string, number>;
  metrics: Record<string, any>;
}

async function handle<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try {
      const body = await r.json();
      if (body.detail) msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error, e.g. Vercel 413 */
      if (r.status === 413) msg = "File too large for the server. Try a shorter time range.";
    }
    throw new Error(msg);
  }
  return r.json();
}

export const api = {
  overview: () => fetch(`${BASE}/api/overview`).then((r) => handle<Overview>(r)),
  models: () => fetch(`${BASE}/api/models`).then((r) => handle<ModelInfo[]>(r)),
  sample: (model: ModelKey | "all", offset_hours: number, hours: number) =>
    fetch(`${BASE}/api/analyze/sample`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, offset_hours, hours }),
    }).then((r) => handle<AnalysisResponse>(r)),
  upload: (file: File, model: ModelKey | "all") => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("model", model);
    return fetch(`${BASE}/api/analyze/upload`, { method: "POST", body: fd }).then((r) =>
      handle<AnalysisResponse>(r),
    );
  },
  url: (path: string) => `${BASE}${path}`,
};

export const fmtTs = (iso: string, withYear = false) => {
  const d = new Date(iso.replace(" ", "T"));
  if (isNaN(d.getTime())) return iso;
  return d.toLocaleString("en-GB", {
    timeZone: "UTC",
    day: "2-digit",
    month: "short",
    ...(withYear ? { year: "numeric" } : {}),
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
};

export const fmtDuration = (min: number) => {
  if (min < 60) return `${min} min`;
  const h = Math.floor(min / 60);
  const m = min % 60;
  if (h < 48) return m ? `${h} h ${m} min` : `${h} h`;
  return `${(min / 1440).toFixed(1)} days`;
};
