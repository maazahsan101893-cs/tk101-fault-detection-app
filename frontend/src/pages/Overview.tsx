import { Bar, BarChart, CartesianGrid, Cell, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { fmtTs, MODEL_COLORS, Overview as OverviewData } from "../api";
import { tooltipStyle } from "../components/chartTheme";

const pct = (v: number) => `${(v * 100).toFixed(2)}%`;
const isPhysics = (f: string) => /^(ERR_|RES_|UA_|VALVE_|DIFF_)/.test(f) || f.endsWith("_ROLL_STD");

export default function Overview({ data, onAnalyze }: { data: OverviewData | null; onAnalyze: () => void }) {
  if (!data) return <div className="empty">Loading project results…</div>;
  const best = data.leaderboard.find((r) => r.is_best) ?? data.leaderboard[0];
  const gd = data.gradient_descent.gradient_descent_comparison;
  const splits = data.splits.splits;

  return (
    <div className="stack">
      <section className="panel hero">
        <div>
          <span className="pill">90 days · 129,600 one-minute samples · no fault labels</span>
          <h2 style={{ marginTop: 12 }}>
            Detects abnormal operation in the TK-101 hold tank and points to the tags driving it.
          </h2>
          <p>
            The pipeline adds first-principles physics features (loop tracking errors, energy-balance residuals,
            coil E-101 fouling, valve stiction, steam-trap subcooling) to the historian tags. Unsupervised
            detectors are then fitted on the first 70% of the timeline. Alarm limits are set at the 99th
            percentile of training scores, so the baseline false-alarm rate is 1%.
          </p>
          <p>
            Cascade control: <span className="mono">TIC-101 → FIC-201</span> (tank temperature → steam flow) and{" "}
            <span className="mono">LIC-101</span> (inventory level).
          </p>
          <div className="row" style={{ marginTop: 16 }}>
            <button className="btn" onClick={onAnalyze}>
              Run fault detection
            </button>
          </div>
        </div>
        <div className="grid g2">
          <div className="kpi alert">
            <div className="label">Best model</div>
            <div className="value" style={{ fontSize: 18 }}>{best.display_name.split("(")[0]}</div>
            <div className="hint">Composite {best.composite_score} / 100</div>
          </div>
          <div className="kpi">
            <div className="label">Test F1</div>
            <div className="value">{best.test_f1.toFixed(3)}</div>
            <div className="hint">ROC-AUC {best.test_roc_auc.toFixed(3)}</div>
          </div>
          <div className="kpi">
            <div className="label">Test precision</div>
            <div className="value">{pct(best.test_precision)}</div>
            <div className="hint">Recall {pct(best.test_recall)}</div>
          </div>
          <div className="kpi">
            <div className="label">Fault events (test)</div>
            <div className="value">{best.test_events}</div>
            <div className="hint">After spike filter &amp; gap merge</div>
          </div>
        </div>
      </section>

      <section className="panel">
        <h2>Model leaderboard</h2>
        <p className="sub">
          All models trained on the 70% split, tuned on validation and scored on the unseen 15% test split.
          Composite = 35% ROC-AUC + 30% F1 + 15% false-alarm-rate calibration + 10% interpretability + 10% stability.
        </p>
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th>#</th>
                <th>Model</th>
                <th className="num">Train RMSE</th>
                <th className="num">Val RMSE</th>
                <th className="num">Test RMSE</th>
                <th className="num">Test ROC-AUC</th>
                <th className="num">Test Precision</th>
                <th className="num">Test Recall</th>
                <th className="num">Test F1</th>
                <th className="num">Accuracy</th>
                <th className="num">Composite</th>
              </tr>
            </thead>
            <tbody>
              {data.leaderboard.map((r) => (
                <tr key={r.model_key}>
                  <td className="mono">{r.rank}</td>
                  <td>
                    <div style={{ fontWeight: 500 }}>
                      {r.display_name} {r.is_best && <span className="pill">winner</span>}
                    </div>
                    <div className="muted" style={{ fontSize: 12 }}>
                      {r.family} · {r.best_epoch_or_stage}
                    </div>
                  </td>
                  <td className="num">{r.train_rmse.toFixed(4)}</td>
                  <td className="num">{r.val_rmse.toFixed(4)}</td>
                  <td className="num">{r.test_rmse.toFixed(4)}</td>
                  <td className="num">{r.test_roc_auc.toFixed(4)}</td>
                  <td className="num">{r.test_precision.toFixed(4)}</td>
                  <td className="num">{r.test_recall.toFixed(4)}</td>
                  <td className="num">{r.test_f1.toFixed(4)}</td>
                  <td className="num">{pct(r.test_accuracy)}</td>
                  <td className="num" style={{ color: r.is_best ? "var(--amber)" : undefined }}>
                    {r.composite_score.toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <div className="grid g2">
        <section className="panel">
          <h2>Composite score by model</h2>
          <p className="sub">Out of 100, on the held-out test split.</p>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={data.leaderboard} layout="vertical" margin={{ left: 20, right: 30 }}>
              <CartesianGrid stroke="#243041" horizontal={false} />
              <XAxis type="number" domain={[0, 100]} />
              <YAxis
                type="category"
                dataKey="display_name"
                width={150}
                tickFormatter={(v: string) => v.split("(")[0].trim()}
              />
              <Tooltip {...tooltipStyle} formatter={(v: number) => v.toFixed(2)} />
              <Bar dataKey="composite_score" name="Composite" radius={[0, 4, 4, 0]}>
                {data.leaderboard.map((r) => (
                  <Cell key={r.model_key} fill={MODEL_COLORS[r.model_key]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </section>

        <section className="panel">
          <h2>Chronological 70 / 15 / 15 split</h2>
          <p className="sub">
            {data.splits.total_rows.toLocaleString()} rows × {data.splits.columns} columns. Imputation medians,
            scaler and control limits are fitted on training data only, so no future data leaks backwards.
          </p>
          <div style={{ display: "flex", height: 34, borderRadius: 8, overflow: "hidden", margin: "8px 0 16px" }}>
            {Object.entries(splits).map(([k, s], i) => (
              <div
                key={k}
                style={{
                  flex: s.ratio,
                  background: ["#1f2a37", "#2b3a4d", "rgba(245,158,11,0.35)"][i],
                  display: "grid",
                  placeItems: "center",
                  fontSize: 12,
                  fontFamily: "var(--mono)",
                }}
              >
                {k} {Math.round(s.ratio * 100)}%
              </div>
            ))}
          </div>
          <table className="data">
            <tbody>
              {Object.entries(splits).map(([k, s]) => (
                <tr key={k}>
                  <td style={{ textTransform: "capitalize" }}>{k}</td>
                  <td className="num">{s.rows.toLocaleString()} rows</td>
                  <td className="num muted">
                    {fmtTs(s.start_timestamp)} → {fmtTs(s.end_timestamp)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </div>

      <section className="panel">
        <h2>Gradient descent study (LSTM autoencoder)</h2>
        <p className="sub">
          Same network trained three ways. Mini-batches of 256 average out single-minute sensor noise while still
          updating weights many times per epoch, giving the lowest cost and the best test F1.
        </p>
        <div className="grid" style={{ gridTemplateColumns: "minmax(0,1.5fr) minmax(0,1fr)" }}>
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th>Optimizer</th>
                  <th className="num">Updates / epoch</th>
                  <th className="num">Train cost (MSE)</th>
                  <th className="num">Test RMSE</th>
                  <th className="num">Test F1</th>
                  <th className="num">Score</th>
                </tr>
              </thead>
              <tbody>
                {gd.map((g) => (
                  <tr key={g.optimizer_key}>
                    <td>
                      <span style={{ whiteSpace: "nowrap" }}>{g.optimizer_name.replace(/ \(.*\)/, "")}</span>
                      <div className="muted" style={{ fontSize: 12 }}>{g.batch_size}</div>
                    </td>
                    <td className="num">{g.updates_per_epoch.toLocaleString()}</td>
                    <td className="num">{g.train_cost_function_mse.toFixed(4)}</td>
                    <td className="num">{g.test_rmse.toFixed(4)}</td>
                    <td className="num">{g.test_f1_score.toFixed(4)}</td>
                    <td className="num">{g.composite_score.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart
              data={gd.map((g) => ({ name: g.optimizer_name.match(/\((.*)\)/)?.[1] ?? g.optimizer_name, f1: g.test_f1_score, cost: g.train_cost_function_mse }))}
              margin={{ right: 10 }}
            >
              <CartesianGrid stroke="#243041" vertical={false} />
              <XAxis dataKey="name" />
              <YAxis domain={[0, 1.1]} />
              <Tooltip {...tooltipStyle} formatter={(v: number) => v.toFixed(4)} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey="f1" name="Test F1" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              <Bar dataKey="cost" name="Train cost (MSE)" fill="#38bdf8" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="panel">
        <h2>Model inputs: {data.features.length} selected features</h2>
        <p className="sub">
          90 raw tags + 18 physics features → drop controller modes, collinear tags and uncoupled utility skids
          (CW, N2, SD, AIR).
        </p>
        <div className="chips">
          {data.features.map((f) => (
            <span className="chip" key={f} style={isPhysics(f) ? { borderColor: "rgba(245,158,11,.55)", color: "var(--amber)" } : undefined}>
              {f}
            </span>
          ))}
        </div>
        <p className="sub" style={{ marginTop: 12, marginBottom: 0 }}>
          <span style={{ color: "var(--amber)" }}>Amber</span> = engineered physics feature.
        </p>
      </section>
    </div>
  );
}
