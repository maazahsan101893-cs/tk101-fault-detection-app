"""
Analysis service: loads artifacts once, turns raw historian rows into
anomaly scores, consolidated fault events and root-cause contributions.
"""
import csv
import io
import json
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np

from .detectors import Detector, load_detectors
from .features import FEATURE_DESCRIPTIONS, UA_WINDOW, engineer_features, impute

ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", Path(__file__).resolve().parent.parent / "artifacts"))
MAX_CHART_POINTS = 1500
MIN_SPIKE_MIN = 2  # alarms of <= 2 min are discarded
MERGE_GAP_MIN = 15  # gaps of <= 15 min between alarms are merged
RAW_TAGS_NEEDED_FOR_FEATURES = {"TIC-101.SP", "LIC-101.SP", "CALC-DUTY-PROC", "TI-100"}


class InputError(ValueError):
    """Raised for user-fixable problems with uploaded data."""


def consolidate_events(flags: np.ndarray, min_spike: int = MIN_SPIKE_MIN, merge_gap: int = MERGE_GAP_MIN):
    f = flags.astype(bool).copy()
    n = len(f)
    i = 0
    while i < n:  # drop short spikes
        if f[i]:
            j = i
            while j < n and f[j]:
                j += 1
            if j - i <= min_spike:
                f[i:j] = False
            i = j
        else:
            i += 1
    i = 0
    while i < n:  # merge short recovery gaps
        if not f[i]:
            j = i
            while j < n and not f[j]:
                j += 1
            if i > 0 and j < n and j - i <= merge_gap:
                f[i:j] = True
            i = j
        else:
            i += 1
    spans = []
    i = 0
    while i < n:
        if f[i]:
            j = i
            while j < n and f[j]:
                j += 1
            spans.append((i, j - 1))
            i = j
        else:
            i += 1
    return f, spans


def downsample(timestamps: list[str], scores: np.ndarray, flags: np.ndarray, max_points: int = MAX_CHART_POINTS):
    """Bucket the series, keeping each bucket's peak so short spikes stay visible."""
    n = len(scores)
    step = max(1, int(np.ceil(n / max_points)))
    out = []
    for b in range(0, n, step):
        seg = scores[b : b + step]
        k = int(np.argmax(seg))
        out.append(
            {
                "t": timestamps[b + k],
                "score": round(float(seg[k]), 5),
                "event": bool(flags[b : b + step].any()),
            }
        )
    return out, step


class Engine:
    def __init__(self, artifacts: Path = ARTIFACTS):
        self.artifacts = artifacts
        self.detectors: dict[str, Detector] = load_detectors(artifacts / "models")
        w = np.load(artifacts / "test_window.npz", allow_pickle=False)
        self.tag_names = [str(t) for t in w["tag_names"]]
        self.warmup_values = w["warmup_values"]
        self.warmup_ts = [str(t) for t in w["warmup_timestamps"]]
        self.values = w["values"]
        self.timestamps = [str(t) for t in w["timestamps"]]
        self.results = self._load_results()
        any_det = next(iter(self.detectors.values()))
        self.required_tags = sorted(
            {f for f in any_det.features if f in self.tag_names} | RAW_TAGS_NEEDED_FOR_FEATURES
        )

    # ---------- static results from the training repo ----------
    def _load_results(self) -> dict:
        r = self.artifacts / "results"
        out = {"models": {}}
        for key in self.detectors:
            out["models"][key] = json.loads((r / f"{key}_scores.json").read_text())
        out["manifest"] = json.loads((r / "checkpoints_manifest.json").read_text())
        out["splits"] = json.loads((r / "split_metadata.json").read_text())
        out["gradient_descent"] = json.loads((r / "gradient_descent_comparison.json").read_text())
        return out

    # ---------- core analysis ----------
    def _scaled(self, det: Detector, raw: np.ndarray) -> np.ndarray:
        X_sel, names = engineer_features(impute(raw, det.train_medians), self.tag_names)
        if names != det.features:
            raise RuntimeError(f"Feature mismatch for {det.key}")
        return det.scale(X_sel)

    def analyze(self, raw: np.ndarray, timestamps: list[str], warmup: int, model_keys: list[str], warnings=None) -> dict:
        """raw: (warmup + n, 90) raw tag values in self.tag_names order. Rows before `warmup` are context."""
        n = len(timestamps)
        assert raw.shape[0] == warmup + n
        results = []
        cache: dict[bytes, np.ndarray] = {}
        for key in model_keys:
            det = self.detectors[key]
            ck = det.train_medians.tobytes() + det.scaler_mean.tobytes()
            if ck not in cache:
                cache[ck] = self._scaled(det, raw)
            Xs = cache[ck]
            scores = det.score(Xs, warmup)
            raw_flags = scores > det.threshold
            clean, spans = consolidate_events(raw_flags)

            events = []
            overall = np.zeros(len(det.features))
            for i, (s, e) in enumerate(spans, start=1):
                contrib = det.contributions(Xs, warmup + s, warmup + e + 1)
                overall += contrib * (e - s + 1)
                events.append(
                    {
                        "event_id": i,
                        "start": timestamps[s],
                        "end": timestamps[e],
                        "start_idx": s,
                        "end_idx": e,
                        "duration_min": e - s + 1,
                        "peak_score": round(float(scores[s : e + 1].max()), 4),
                        "mean_score": round(float(scores[s : e + 1].mean()), 4),
                        "contributors": self._top(det, contrib, 10),
                    }
                )
            series, step = downsample(timestamps, scores, clean)
            results.append(
                {
                    "model": key,
                    "model_name": det.name,
                    "contribution_method": det.contribution_method,
                    "threshold": round(det.threshold, 5),
                    "summary": {
                        "rows": n,
                        "start": timestamps[0],
                        "end": timestamps[-1],
                        "raw_alarm_pct": round(float(raw_flags.mean() * 100), 2),
                        "event_pct": round(float(clean.mean() * 100), 2),
                        "minutes_in_fault": int(clean.sum()),
                        "events": len(events),
                        "mean_score": round(float(scores.mean()), 4),
                        "max_score": round(float(scores.max()), 4),
                    },
                    "overall_contributors": self._top(det, overall, 10) if spans else [],
                    "events": events,
                    "series": series,
                    "series_bucket_min": step,
                }
            )
        return {"results": results, "warnings": warnings or []}

    @staticmethod
    def _top(det: Detector, contrib: np.ndarray, k: int) -> list[dict]:
        total = float(contrib.sum()) or 1.0
        order = np.argsort(contrib)[::-1][:k]
        return [
            {
                "feature": det.features[i],
                "description": FEATURE_DESCRIPTIONS.get(det.features[i], det.features[i]),
                "value": round(float(contrib[i]), 4),
                "share_pct": round(float(contrib[i]) / total * 100, 2),
            }
            for i in order
        ]

    # ---------- bundled test window ----------
    def window_info(self) -> dict:
        return {
            "start": self.timestamps[0],
            "end": self.timestamps[-1],
            "rows": len(self.timestamps),
            "hours": len(self.timestamps) / 60,
        }

    def analyze_sample(self, offset_hours: float, hours: float, model_keys: tuple[str, ...]) -> dict:
        return _cached_sample(self, round(offset_hours, 2), round(hours, 2), model_keys)

    def _analyze_sample(self, offset_hours: float, hours: float, model_keys: tuple[str, ...]) -> dict:
        total = len(self.timestamps)
        start = int(round(offset_hours * 60))
        rows = int(round(hours * 60))
        if not (0 <= start < total) or rows < 30:
            raise InputError("Window must start inside the test period and be at least 30 minutes long.")
        stop = min(total, start + rows)
        full = np.vstack([self.warmup_values, self.values])
        abs_start = len(self.warmup_values) + start
        ctx_start = max(0, abs_start - UA_WINDOW)
        raw = full[ctx_start : len(self.warmup_values) + stop]
        return self.analyze(raw, self.timestamps[start:stop], abs_start - ctx_start, list(model_keys))

    def sample_csv(self, offset_hours: float = 126, hours: float = 24) -> str:
        start = int(offset_hours * 60)
        stop = min(len(self.timestamps), start + int(hours * 60))
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["timestamp", *self.tag_names])
        for i in range(start, stop):
            w.writerow([self.timestamps[i], *("" if np.isnan(v) else f"{v:.6g}" for v in self.values[i])])
        return buf.getvalue()

    def template_csv(self) -> str:
        return "timestamp," + ",".join(self.tag_names) + "\n"

    # ---------- uploads ----------
    def parse_upload(self, content: bytes, max_rows: int = 50_000) -> tuple[np.ndarray, list[str], list[str]]:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise InputError("File must be UTF-8 encoded CSV.") from exc
        reader = csv.reader(io.StringIO(text))
        try:
            header = [h.strip() for h in next(reader)]
        except StopIteration as exc:
            raise InputError("The file is empty.") from exc
        if not header or header[0].lower() not in {"timestamp", "time", "datetime"}:
            raise InputError("The first column must be 'timestamp'.")
        col_of = {h: i for i, h in enumerate(header)}
        missing = [t for t in self.required_tags if t not in col_of]
        if missing:
            raise InputError(
                f"Missing {len(missing)} required TK-101 tag column(s): {', '.join(missing[:12])}"
                + (" ..." if len(missing) > 12 else "")
            )
        src_idx = [col_of.get(t) for t in self.tag_names]
        ts, rows = [], []
        for line_no, row in enumerate(reader, start=2):
            if not row or not row[0].strip():
                continue
            if len(rows) >= max_rows:
                raise InputError(f"Too many rows (limit {max_rows:,}).")
            vals = []
            for j in src_idx:
                cell = row[j].strip() if j is not None and j < len(row) else ""
                try:
                    vals.append(float(cell) if cell else np.nan)
                except ValueError as exc:
                    raise InputError(f"Non-numeric value '{cell}' on line {line_no}.") from exc
            ts.append(row[0].strip())
            rows.append(vals)
        if len(rows) < 30:
            raise InputError("Need at least 30 rows (30 minutes of 1-minute data).")

        parsed = []
        for t in ts:
            try:
                parsed.append(datetime.fromisoformat(t.replace("Z", "+00:00")))
            except ValueError as exc:
                raise InputError(f"Unrecognised timestamp '{t}'. Use ISO format, e.g. 2024-05-20 06:00:00+00:00.") from exc
        X = np.array(rows, dtype=np.float64)
        warnings = []
        order = sorted(range(len(parsed)), key=parsed.__getitem__)
        if order != list(range(len(parsed))):
            X, ts, parsed = X[order], [ts[i] for i in order], [parsed[i] for i in order]
            warnings.append("Rows were not in time order and have been sorted.")
        try:
            gaps = np.diff([p.timestamp() for p in parsed])
            med = float(np.median(gaps))
            if abs(med - 60) > 1:
                warnings.append(f"Median sample interval is {med:.0f}s; the models were trained on 60s data.")
        except TypeError:
            warnings.append("Timestamps mix time-zone aware and naive values; interval check skipped.")
        optional_missing = [t for t, j in zip(self.tag_names, src_idx) if j is None]
        if optional_missing:
            warnings.append(
                f"{len(optional_missing)} non-TK-101 column(s) not provided (utility skids). They are not used by the models."
            )
        if len(ts) < UA_WINDOW:
            warnings.append(
                "Less than 24 h of data: the coil fouling feature (UA_ROLLING_DIFF) uses a shorter baseline, so early scores are approximate."
            )
        nan_pct = float(np.isnan(X[:, [i for i, j in enumerate(src_idx) if j is not None]]).mean() * 100)
        if nan_pct > 0:
            warnings.append(f"{nan_pct:.2f}% of cells were empty and were filled with training medians.")
        return X, ts, warnings


@lru_cache(maxsize=64)
def _cached_sample(engine: Engine, offset_hours: float, hours: float, model_keys: tuple[str, ...]) -> dict:
    return engine._analyze_sample(offset_hours, hours, model_keys)
