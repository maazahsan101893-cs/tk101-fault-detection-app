import { useEffect, useRef, useState } from "react";
import {
  AnalysisResponse,
  api,
  FaultEvent,
  fmtDuration,
  fmtTs,
  MODEL_COLORS,
  MODEL_LABELS,
  ModelKey,
  ModelResult,
  Overview,
} from "../api";
import ScoreChart from "../components/ScoreChart";
import Contributors from "../components/Contributors";

type Source = "test" | "upload";
type ModelChoice = ModelKey | "all";

const MAX_UPLOAD_MB = Number(import.meta.env.VITE_MAX_UPLOAD_MB ?? 4);
const TEST_START = Date.parse("2024-05-16T18:00:00Z");
const DURATIONS = [6, 24, 72, 168, 324];

const hoursToTs = (h: number) => new Date(TEST_START + h * 3600_000).toISOString();

export default function Analyze({ overview }: { overview: Overview | null }) {
  const totalHours = overview?.test_window.hours ?? 324;
  const [source, setSource] = useState<Source>("test");
  const [model, setModel] = useState<ModelChoice>("pca_t2_q");
  const [offset, setOffset] = useState(0);
  const [hours, setHours] = useState(324);
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [resp, setResp] = useState<AnalysisResponse | null>(null);
  const [logScale, setLogScale] = useState(true);
  const fileRef = useRef<HTMLInputElement>(null);
  const autoRan = useRef(false);

  const maxOffset = Math.max(0, totalHours - Math.min(hours, totalHours));
  useEffect(() => {
    if (offset > maxOffset) setOffset(maxOffset);
  }, [maxOffset, offset]);

  const run = async () => {
    setError(null);
    setLoading(true);
    try {
      if (source === "upload") {
        if (!file) throw new Error("Choose a CSV file first.");
        if (file.size > MAX_UPLOAD_MB * 1024 * 1024)
          throw new Error(
            `File is ${(file.size / 1048576).toFixed(1)} MB; the hosted API accepts up to ${MAX_UPLOAD_MB} MB (roughly 4–5 days of 1-minute data). Split the file or run the Docker version for larger files.`,
          );
        setResp(await api.upload(file, model));
      } else {
        setResp(await api.sample(model, offset, Math.min(hours, totalHours)));
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  // Show something useful immediately on first visit
  useEffect(() => {
    if (!autoRan.current) {
      autoRan.current = true;
      run();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="stack">
      <section className="panel">
        <div className="controls">
          <div className="stack" style={{ gap: 14 }}>
            <div className="field">
              <label>Data source</label>
              <div className="seg">
                <button className={source === "test" ? "on" : ""} onClick={() => setSource("test")}>
                  Held-out test split
                </button>
                <button className={source === "upload" ? "on" : ""} onClick={() => setSource("upload")}>
                  Upload historian CSV
                </button>
              </div>
            </div>
            <div className="field">
              <label>Detector</label>
              <div className="seg">
                {(Object.keys(MODEL_LABELS) as ModelKey[]).map((k) => (
                  <button key={k} className={model === k ? "on" : ""} onClick={() => setModel(k)}>
                    {MODEL_LABELS[k]}
                    {k === overview?.best_model && <span style={{ color: "var(--amber)" }}> ★</span>}
                  </button>
                ))}
                <button className={model === "all" ? "on" : ""} onClick={() => setModel("all")}>
                  Compare all
                </button>
              </div>
            </div>
          </div>

          <div className="stack" style={{ gap: 14 }}>
            {source === "test" ? (
              <>
                <div className="field">
                  <label>Window length</label>
                  <div className="seg">
                    {DURATIONS.map((d) => (
                      <button key={d} className={hours === d ? "on" : ""} onClick={() => setHours(d)}>
                        {d === 324 ? "Full 13.5 d" : d >= 24 ? `${d / 24} d` : `${d} h`}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="field">
                  <label>Window start</label>
                  <input
                    type="range"
                    min={0}
                    max={maxOffset}
                    step={1}
                    value={offset}
                    disabled={maxOffset === 0}
                    onChange={(e) => setOffset(Number(e.target.value))}
                  />
                  <div className="range-row">
                    <span>{fmtTs(hoursToTs(offset), true)}</span>
                    <span>→ {fmtTs(hoursToTs(Math.min(offset + hours, totalHours)), true)} UTC</span>
                  </div>
                </div>
              </>
            ) : (
              <div className="field">
                <label>Historian CSV (1-minute rows, same tag names as historian_60s.csv)</label>
                <div
                  className={`drop ${file ? "has" : ""}`}
                  onClick={() => fileRef.current?.click()}
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => {
                    e.preventDefault();
                    if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
                  }}
                >
                  {file ? (
                    <>
                      <span className="mono">{file.name}</span> · {(file.size / 1048576).toFixed(2)} MB
                    </>
                  ) : (
                    "Drop a CSV here or click to browse"
                  )}
                </div>
                <input
                  ref={fileRef}
                  type="file"
                  accept=".csv,text/csv"
                  hidden
                  onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                />
                <div className="row muted" style={{ fontSize: 12, marginTop: 8 }}>
                  <a href={api.url("/api/sample.csv")}>Download a 24 h sample</a>
                  <span>·</span>
                  <a href={api.url("/api/template.csv")}>Blank template</a>
                  <span>· max {MAX_UPLOAD_MB} MB</span>
                </div>
              </div>
            )}
            <div className="row">
              <button className="btn" onClick={run} disabled={loading}>
                {loading && <span className="spinner" />}
                {loading ? "Analysing…" : "Run detection"}
              </button>
              <span className="spacer" />
              <label className="row muted" style={{ gap: 6, fontSize: 12, cursor: "pointer" }}>
                <input type="checkbox" checked={logScale} onChange={(e) => setLogScale(e.target.checked)} /> Log scale
              </label>
            </div>
          </div>
        </div>
      </section>

      {error && <div className="notice error">{error}</div>}
      {resp?.warnings.map((w) => (
        <div className="notice" key={w}>
          {w}
        </div>
      ))}

      {resp &&
        (resp.results.length === 1 ? (
          <SingleResult key={resp.results[0].model + resp.results[0].summary.start} result={resp.results[0]} logScale={logScale} />
        ) : (
          <CompareResults results={resp.results} logScale={logScale} />
        ))}
    </div>
  );
}

function SummaryTiles({ r }: { r: ModelResult }) {
  const s = r.summary;
  return (
    <div className="grid g4">
      <div className={`kpi ${s.events ? "alert" : ""}`}>
        <div className="label">Fault events</div>
        <div className="value">{s.events}</div>
        <div className="hint">{fmtDuration(s.minutes_in_fault)} in fault</div>
      </div>
      <div className="kpi">
        <div className="label">Time in fault</div>
        <div className="value">{s.event_pct.toFixed(1)}%</div>
        <div className="hint">raw alarms {s.raw_alarm_pct.toFixed(1)}%</div>
      </div>
      <div className="kpi">
        <div className="label">Peak score</div>
        <div className="value">{s.max_score.toFixed(2)}</div>
        <div className="hint">{(s.max_score / r.threshold).toFixed(1)}× the alarm limit</div>
      </div>
      <div className="kpi">
        <div className="label">Window</div>
        <div className="value" style={{ fontSize: 18 }}>{fmtDuration(s.rows)}</div>
        <div className="hint">{s.rows.toLocaleString()} samples</div>
      </div>
    </div>
  );
}

function SingleResult({ result, logScale }: { result: ModelResult; logScale: boolean }) {
  const [selected, setSelected] = useState<FaultEvent | null>(null);
  const color = MODEL_COLORS[result.model];
  return (
    <div className="stack">
      <SummaryTiles r={result} />
      <section className="panel">
        <div className="row" style={{ marginBottom: 4 }}>
          <h2>Anomaly score · {result.model_name}</h2>
          <span className="spacer" />
          <span className="muted" style={{ fontSize: 12 }}>
            {fmtTs(result.summary.start, true)} → {fmtTs(result.summary.end, true)} UTC
            {result.series_bucket_min > 1 && ` · peak per ${result.series_bucket_min} min`}
          </span>
        </div>
        <p className="sub">
          Red bands are consolidated fault events (alarms of 2 min or less removed, gaps of 15 min or less merged).
          Dashed line is the 99th-percentile training limit. Click an event below to highlight it.
        </p>
        <ScoreChart result={result} color={color} logScale={logScale} selected={selected} />
      </section>

      <div className="grid g3">
        <section className="panel">
          <h2>Fault events</h2>
          <p className="sub">Top tags by {result.contribution_method.toLowerCase()} during each event.</p>
          {result.events.length === 0 ? (
            <div className="empty">No fault events: the tank stayed inside its normal operating envelope.</div>
          ) : (
            <div className="table-wrap" style={{ maxHeight: 420, overflowY: "auto" }}>
              <table className="data">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Start (UTC)</th>
                    <th className="num">Duration</th>
                    <th className="num">Peak</th>
                    <th>Root-cause tags</th>
                  </tr>
                </thead>
                <tbody>
                  {result.events.map((e) => (
                    <tr
                      key={e.event_id}
                      className={`clickable ${selected?.event_id === e.event_id ? "selected" : ""}`}
                      onClick={() => setSelected(selected?.event_id === e.event_id ? null : e)}
                    >
                      <td className="mono">{e.event_id}</td>
                      <td className="mono">{fmtTs(e.start)}</td>
                      <td className="num">{fmtDuration(e.duration_min)}</td>
                      <td className="num">{e.peak_score.toFixed(2)}</td>
                      <td>
                        <div className="chips">
                          {e.contributors.slice(0, 3).map((c) => (
                            <span className="chip" key={c.feature}>{c.feature}</span>
                          ))}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
        <section className="panel">
          <h2>{selected ? `Event ${selected.event_id} root cause` : "What drives the alarms"}</h2>
          <p className="sub">
            {selected
              ? `${fmtTs(selected.start)} → ${fmtTs(selected.end)} UTC · ${fmtDuration(selected.duration_min)}`
              : "Share of contribution across all fault events in this window (weighted by duration)."}
          </p>
          <Contributors items={selected ? selected.contributors : result.overall_contributors} />
        </section>
      </div>
    </div>
  );
}

function CompareResults({ results, logScale }: { results: ModelResult[]; logScale: boolean }) {
  return (
    <div className="stack">
      <section className="panel">
        <h2>Detector comparison on this window</h2>
        <p className="sub">Same data, four detectors. Each uses its own 99th-percentile training limit.</p>
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th>Detector</th>
                <th className="num">Events</th>
                <th className="num">Time in fault</th>
                <th className="num">Raw alarms</th>
                <th className="num">Peak / limit</th>
                <th>Top root-cause tags</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r) => (
                <tr key={r.model}>
                  <td>
                    <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: MODEL_COLORS[r.model], marginRight: 8 }} />
                    {MODEL_LABELS[r.model]}
                  </td>
                  <td className="num">{r.summary.events}</td>
                  <td className="num">{r.summary.event_pct.toFixed(1)}%</td>
                  <td className="num">{r.summary.raw_alarm_pct.toFixed(1)}%</td>
                  <td className="num">{(r.summary.max_score / r.threshold).toFixed(1)}×</td>
                  <td>
                    <div className="chips">
                      {r.overall_contributors.slice(0, 3).map((c) => (
                        <span className="chip" key={c.feature}>{c.feature}</span>
                      ))}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <div className="grid g2">
        {results.map((r) => (
          <section className="panel" key={r.model}>
            <div className="row">
              <h2>{MODEL_LABELS[r.model]}</h2>
              <span className="spacer" />
              <span className="pill grey">{r.summary.events} events</span>
            </div>
            <ScoreChart result={r} color={MODEL_COLORS[r.model]} logScale={logScale} height={200} compact />
          </section>
        ))}
      </div>
    </div>
  );
}
