import { useEffect, useState } from "react";
import { api, MODEL_COLORS, ModelInfo } from "../api";

const SPLITS: [string, string][] = [
  ["training_split_70pct", "Train 70%"],
  ["validation_split_15pct", "Val 15%"],
  ["testing_split_15pct", "Test 15%"],
];

const ROWS: [string, string, (v: number) => string][] = [
  ["loss_single_sample_median", "Loss (median)", (v) => v.toFixed(5)],
  ["cost_function_mse", "Cost J(θ) (MSE)", (v) => v.toFixed(5)],
  ["rmse", "RMSE", (v) => v.toFixed(4)],
  ["roc_auc", "ROC-AUC", (v) => v.toFixed(4)],
  ["precision", "Precision", (v) => v.toFixed(4)],
  ["recall", "Recall", (v) => v.toFixed(4)],
  ["f1_score", "F1", (v) => v.toFixed(4)],
  ["accuracy", "Accuracy", (v) => `${(v * 100).toFixed(2)}%`],
  ["candidate_fault_events", "Fault events", (v) => String(v)],
];

const ABOUT: Record<string, string> = {
  pca_t2_q:
    "Projects the 40 scaled features onto the principal subspace. Hotelling's T² catches shifts inside the normal correlation structure (steam header swings, gradual fouling). The Q-statistic (SPE) catches broken correlations (valve stiction, steam-trap blow-through). Root-cause contributions come straight from the math, so it is the most interpretable detector.",
  lstm_autoencoder:
    "Encodes a 10-minute window with an LSTM and reconstructs the latest sample. Large reconstruction error means the recent dynamics don't look like normal operation. Its recurrent memory models the multi-minute thermal and transport lags in the tank.",
  local_outlier_factor:
    "Compares each sample's local density with that of its nearest neighbours in a reference set of normal training states. Strong at spotting operating points that sit in sparse regions.",
  isolation_forest:
    "Random axis-aligned splits isolate unusual points quickly. Very precise but conservative: at a 1% false-alarm limit it only flags extreme spikes and misses collinear mass/energy balance shifts.",
};

export default function Models() {
  const [models, setModels] = useState<ModelInfo[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api.models().then(setModels).catch((e) => setErr(e.message));
  }, []);
  if (err) return <div className="notice error">{err}</div>;
  if (!models) return <div className="empty">Loading models…</div>;

  return (
    <div className="stack">
      {models.map((m) => (
        <section className="panel" key={m.key}>
          <div className="row" style={{ alignItems: "flex-start" }}>
            <span style={{ width: 4, alignSelf: "stretch", borderRadius: 2, background: MODEL_COLORS[m.key] }} />
            <div style={{ flex: 1 }}>
              <div className="row">
                <h2 style={{ fontSize: 16 }}>{m.name}</h2>
                <span className="pill grey">{m.family}</span>
                <span className="spacer" />
                <span className="mono" style={{ color: "var(--amber)", fontSize: 18 }}>
                  {m.metrics.composite_project_score?.toFixed(2)}
                  <span className="muted" style={{ fontSize: 12 }}> / 100</span>
                </span>
              </div>
              <p className="sub" style={{ marginTop: 8 }}>{ABOUT[m.key]}</p>
              <div className="grid g3">
                <div className="table-wrap">
                  <table className="data">
                    <thead>
                      <tr>
                        <th>Metric</th>
                        {SPLITS.map(([, l]) => (
                          <th className="num" key={l}>{l}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {ROWS.map(([k, label, f]) => (
                        <tr key={k}>
                          <td>{label}</td>
                          {SPLITS.map(([s]) => {
                            const v = m.metrics[s]?.[k];
                            return (
                              <td className="num" key={s}>
                                {typeof v === "number" ? f(v) : <span className="muted">n/a</span>}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="stack" style={{ gap: 8 }}>
                  <div className="kpi">
                    <div className="label">Best stage</div>
                    <div className="mono" style={{ marginTop: 6 }}>{m.metrics.best_epoch_or_stage}</div>
                  </div>
                  <div className="kpi">
                    <div className="label">Configuration</div>
                    <table className="data" style={{ marginTop: 4 }}>
                      <tbody>
                        <tr>
                          <td>alarm limit (99%)</td>
                          <td className="num">{m.threshold.toFixed(4)}</td>
                        </tr>
                        {Object.entries(m.config).map(([k, v]) => (
                          <tr key={k}>
                            <td>{k.replace(/_/g, " ")}</td>
                            <td className="num">{v}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      ))}
    </div>
  );
}
