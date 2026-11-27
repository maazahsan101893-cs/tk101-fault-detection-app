"""
Numpy-only inference for the four trained detectors. Each class loads the
.npz checkpoint written by the ChemicalE training scripts and reproduces the
exact scoring function used during evaluation.

Every detector exposes:
    score(Xs, start)        -> anomaly scores for rows Xs[start:]
                               (rows before `start` are history/context only)
    contributions(Xs, s, e) -> per-feature contribution over rows Xs[s:e]
"""
import json
import math
from pathlib import Path

import numpy as np


class Detector:
    key: str = ""
    name: str = ""
    family: str = ""
    contribution_method: str = ""

    def __init__(self, path: Path):
        ck = np.load(path, allow_pickle=False)
        self.scaler_mean = ck["scaler_mean"]
        self.scaler_std = ck["scaler_std"]
        self.train_medians = ck["train_medians"]
        self.features = [str(f) for f in ck["selected_features"]]
        self.threshold = float(ck["score_threshold"])
        self._load(ck)

    def _load(self, ck) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def scale(self, X_sel: np.ndarray) -> np.ndarray:
        return (X_sel - self.scaler_mean) / self.scaler_std

    def score(self, Xs: np.ndarray, start: int) -> np.ndarray:
        raise NotImplementedError

    def contributions(self, Xs: np.ndarray, s: int, e: int) -> np.ndarray:
        raise NotImplementedError

    def config(self) -> dict:
        return {}


class PCAT2Q(Detector):
    key = "pca_t2_q"
    name = "PCA Multivariate Monitor (Hotelling T² + Q)"
    family = "Subspace projection & residual SPC"
    contribution_method = "Hotelling T² + Q-statistic contributions"

    def _load(self, ck) -> None:
        self.P = ck["components"]
        self.lam = ck["eigenvalues"]
        self.k = int(ck["n_components"])
        self.w = float(ck["t2_weight"])
        self.t2_ucl = float(ck["t2_ucl"])
        self.q_ucl = float(ck["q_ucl"])
        self.retained = float(ck["retained_variance"])

    def score(self, Xs, start):
        X = Xs[start:]
        s = X @ self.P
        t2 = np.sum(s**2 / self.lam, axis=1)
        q = np.sum((X - s @ self.P.T) ** 2, axis=1)
        return np.maximum(self.w * t2 / self.t2_ucl, q / self.q_ucl)

    def contributions(self, Xs, s, e):
        X = Xs[s:e]
        recon = (X @ self.P) @ self.P.T
        q_c = (X - recon) ** 2
        t2_w = (self.P / np.sqrt(self.lam)) @ self.P.T
        t2_c = (X @ t2_w) ** 2
        return np.mean(q_c / self.q_ucl + t2_c / self.t2_ucl, axis=0)

    def config(self):
        return {
            "principal_components": self.k,
            "retained_variance_pct": round(self.retained * 100, 2),
            "t2_weight": self.w,
            "t2_ucl": round(self.t2_ucl, 3),
            "q_ucl": round(self.q_ucl, 3),
        }


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))


class LSTMAutoencoder(Detector):
    key = "lstm_autoencoder"
    name = "LSTM Autoencoder"
    family = "Deep temporal sequence reconstruction"
    contribution_method = "Per-tag reconstruction error"

    def _load(self, ck) -> None:
        self.L = int(ck["seq_len"])
        self.H = int(ck["hidden_dim"])
        self.W_enc, self.U_enc, self.b_enc = ck["W_enc"], ck["U_enc"], ck["b_enc"]
        self.W_ctx, self.b_ctx = ck["W_ctx"], ck["b_ctx"]
        self.W_dec, self.b_dec = ck["W_dec"], ck["b_dec"]

    def _windows(self, Xs, start):
        # use real history when available, otherwise repeat the first row (as in training)
        lo = max(0, start - (self.L - 1))
        X = Xs[lo:]
        pad = (self.L - 1) - (start - lo)
        if pad > 0:
            X = np.vstack([np.repeat(X[:1], pad, axis=0), X])
        X = np.ascontiguousarray(X)
        n = len(X) - self.L + 1
        s0, s1 = X.strides
        return np.lib.stride_tricks.as_strided(X, shape=(n, self.L, X.shape[1]), strides=(s0, s0, s1))

    def _forward(self, W):
        B, L, _ = W.shape
        H = self.H
        h = np.zeros((B, H))
        c = np.zeros((B, H))
        h_sum = np.zeros((B, H))
        for t in range(L):
            z = W[:, t, :] @ self.W_enc + h @ self.U_enc + self.b_enc
            f = _sigmoid(z[:, :H])
            i = _sigmoid(z[:, H : 2 * H])
            g = np.tanh(np.clip(z[:, 2 * H : 3 * H], -15, 15))
            o = _sigmoid(z[:, 3 * H :])
            c = f * c + i * g
            h = o * np.tanh(np.clip(c, -15, 15))
            h_sum += h
        latent = np.tanh(np.hstack([h, h_sum / L]) @ self.W_ctx + self.b_ctx)
        return latent @ self.W_dec + self.b_dec

    def _sq_errors(self, Xs, start, stop=None):
        wins = self._windows(Xs, start)
        if stop is not None:
            wins = wins[: stop - start]
        out = np.empty((len(wins), wins.shape[2]))
        for b in range(0, len(wins), 2048):
            Wb = wins[b : b + 2048]
            out[b : b + 2048] = (Wb[:, -1, :] - self._forward(Wb)) ** 2
        return out

    def score(self, Xs, start):
        return np.mean(self._sq_errors(Xs, start), axis=1)

    def contributions(self, Xs, s, e):
        return np.mean(self._sq_errors(Xs, s, e), axis=0)

    def config(self):
        return {"sequence_length_min": self.L, "hidden_units": self.H}


class LocalOutlierFactor(Detector):
    key = "local_outlier_factor"
    name = "Local Outlier Factor (novelty mode)"
    family = "Local density k-NN anomaly detection"
    contribution_method = "Deviation from nearest normal states"

    def _load(self, ck) -> None:
        self.k = int(ck["n_neighbors"])
        self.X_ref = ck["X_ref"]
        self.k_dist_ref = ck["k_dist_ref"]
        self.lrd_ref = ck["lrd_ref"]
        self.ref_norms = np.sum(self.X_ref**2, axis=1)

    def _knn(self, X):
        idx = np.empty((len(X), self.k), dtype=np.int64)
        dist = np.empty((len(X), self.k))
        for b in range(0, len(X), 2048):
            Xb = X[b : b + 2048]
            d = np.sqrt(np.maximum(0.0, np.sum(Xb**2, 1)[:, None] + self.ref_norms[None, :] - 2.0 * Xb @ self.X_ref.T))
            part = np.argpartition(d, self.k - 1, axis=1)[:, : self.k]
            r = np.arange(len(Xb))[:, None]
            dk = d[r, part]
            order = np.argsort(dk, axis=1)
            idx[b : b + 2048] = part[r, order]
            dist[b : b + 2048] = dk[r, order]
        return idx, dist

    def score(self, Xs, start):
        ki, kd = self._knn(Xs[start:])
        reach = np.maximum(kd, self.k_dist_ref[ki])
        lrd_x = 1.0 / (np.mean(reach, axis=1) + 1e-8)
        return np.mean(self.lrd_ref[ki], axis=1) / lrd_x

    def contributions(self, Xs, s, e):
        X = Xs[s:e]
        ki, _ = self._knn(X)
        neigh_mean = self.X_ref[ki].mean(axis=1)
        return np.mean((X - neigh_mean) ** 2, axis=0)

    def config(self):
        return {"neighbors_k": self.k, "reference_states": int(len(self.X_ref))}


def _c_factor(n: int) -> float:
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (math.log(n - 1.0) + 0.5772156649) - 2.0 * (n - 1.0) / n


class IsolationForest(Detector):
    key = "isolation_forest"
    name = "Isolation Forest"
    family = "Tree-based random partitioning"
    contribution_method = "Standardized deviation from training mean"

    def _load(self, ck) -> None:
        self.trees = json.loads(str(ck["trees_json"]))
        self.c_norm = float(ck["c_norm"])
        self.n_estimators = int(ck["n_estimators"])
        self.max_samples = int(ck["max_samples"])

    def _eval(self, node, X, idx, out):
        if len(idx) == 0:
            return
        if node["leaf"]:
            out[idx] = node["depth"] + _c_factor(node["size"])
            return
        left = X[idx, node["feat"]] < node["split"]
        self._eval(node["left"], X, idx[left], out)
        self._eval(node["right"], X, idx[~left], out)

    def score(self, Xs, start):
        X = Xs[start:]
        total = np.zeros(len(X))
        buf = np.zeros(len(X))
        all_idx = np.arange(len(X))
        for tree in self.trees:
            self._eval(tree, X, all_idx, buf)
            total += buf
        return np.power(2.0, -(total / len(self.trees)) / self.c_norm)

    def contributions(self, Xs, s, e):
        return np.mean(Xs[s:e] ** 2, axis=0)

    def config(self):
        return {"trees": self.n_estimators, "max_samples": self.max_samples}


DETECTOR_CLASSES = [PCAT2Q, LSTMAutoencoder, LocalOutlierFactor, IsolationForest]


def load_detectors(models_dir: Path) -> dict[str, Detector]:
    return {cls.key: cls(models_dir / f"{cls.key}.npz") for cls in DETECTOR_CLASSES}
