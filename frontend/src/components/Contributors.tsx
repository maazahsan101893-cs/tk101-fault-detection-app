import { Contributor } from "../api";

export default function Contributors({ items, limit = 8 }: { items: Contributor[]; limit?: number }) {
  if (!items.length) return <div className="empty">No fault events in this window.</div>;
  const top = items.slice(0, limit);
  const max = Math.max(...top.map((c) => c.share_pct));
  return (
    <div className="stack" style={{ gap: 10 }}>
      {top.map((c, i) => (
        <div key={c.feature}>
          <div className="row" style={{ gap: 8, marginBottom: 4 }}>
            <span className="mono muted" style={{ width: 16 }}>{i + 1}</span>
            <span className="mono" style={{ fontWeight: 500 }}>{c.feature}</span>
            <span className="spacer" />
            <span className="mono">{c.share_pct.toFixed(1)}%</span>
          </div>
          <div className="row" style={{ gap: 8 }}>
            <span style={{ width: 16 }} />
            <div style={{ flex: 1 }}>
              <div className="bar">
                <span style={{ width: `${(c.share_pct / max) * 100}%`, opacity: 1 - i * 0.07 }} />
              </div>
              <div className="muted" style={{ fontSize: 12, marginTop: 3 }}>{c.description}</div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
