import { useEffect, useState } from "react";
import { api, Overview as OverviewData } from "./api";
import Overview from "./pages/Overview";
import Analyze from "./pages/Analyze";
import Models from "./pages/Models";

type Tab = "overview" | "analyze" | "models";

const TABS: { key: Tab; label: string }[] = [
  { key: "overview", label: "Overview" },
  { key: "analyze", label: "Fault Detection" },
  { key: "models", label: "Models" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>(() => (location.hash.slice(1) as Tab) || "overview");
  const [overview, setOverview] = useState<OverviewData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.overview().then(setOverview).catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    history.replaceState(null, "", `#${tab}`);
  }, [tab]);

  return (
    <div className="shell">
      <header className="top">
        <div className="brand">
          <div className="brand-mark">
            <svg width="20" height="20" viewBox="0 0 32 32">
              <path d="M2 20h7l3-10 5 15 3-9h10" stroke="#f59e0b" strokeWidth="2.6" fill="none" strokeLinejoin="round" />
            </svg>
          </div>
          <div>
            <h1>
              <span className="tag">TK-101</span> Fault Detection &amp; Root Cause Analysis
            </h1>
            <p>Steam-heated hydrolysis hold tank · historian-based unsupervised monitoring</p>
          </div>
        </div>
        <nav className="tabs">
          {TABS.map((t) => (
            <button key={t.key} className={tab === t.key ? "active" : ""} onClick={() => setTab(t.key)}>
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      {error && (
        <div className="notice error" style={{ marginBottom: 16 }}>
          Could not reach the API: {error}
        </div>
      )}

      {tab === "overview" && <Overview data={overview} onAnalyze={() => setTab("analyze")} />}
      {tab === "analyze" && <Analyze overview={overview} />}
      {tab === "models" && <Models />}

      <footer>
        PIEAS · Department of Chemical Engineering · ML Pipeline Project ·{" "}
        <a href={api.url("/api/docs")} target="_blank" rel="noreferrer">
          API docs
        </a>
      </footer>
    </div>
  );
}
