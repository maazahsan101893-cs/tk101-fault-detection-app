"""
Physics-informed feature engineering, ported 1:1 from
ChemicalE/training/pipeline_utils.py so live scoring matches training.
"""
import numpy as np

UA_WINDOW = 1440  # minutes, coil E-101 fouling baseline
STD_WINDOW = 15  # minutes, sensor volatility

ROLL_STD_TAGS = [
    "TIC-101.PV",
    "TIC-101.OP",
    "FIC-201.PV",
    "FIC-201.OP",
    "LIC-101.PV",
    "LIC-101.OP",
    "PI-201",
    "FI-101",
]

DROPPED_SUFFIXES = (".MODE",)
DROPPED_PREFIXES = ("CW-", "N2-", "SD-", "AIR-")
DROPPED_EXACT = {"CALC-DUTY-PROC", "TI-100", "TIC-101.SP", "LIC-101.SP", "TI-901"}

# Plain-language descriptions shown next to root-cause tags in the UI
FEATURE_DESCRIPTIONS = {
    "TIC-101.PV": "Tank temperature (master loop PV)",
    "TIC-101.OP": "Tank temperature controller output",
    "FIC-201.PV": "Steam flow (slave loop PV)",
    "FIC-201.SP": "Steam flow setpoint (cascaded from TIC-101)",
    "FIC-201.OP": "Steam flow controller output",
    "LIC-101.PV": "Tank level",
    "LIC-101.OP": "Level controller output",
    "TV-101.POS": "Steam valve TV-101 position",
    "LV-101.POS": "Level valve LV-101 position",
    "FI-101": "Feed inflow",
    "FI-102": "Product outflow",
    "TI-102": "Outlet temperature",
    "LI-102": "Secondary level indicator",
    "PI-201": "Steam header pressure",
    "TI-201": "Steam supply temperature",
    "TI-202": "Condensate temperature",
    "AI-101": "Process analyzer",
    "CALC-DUTY": "Steam-side heat duty",
    "CALC-EBAL": "Energy balance residual",
    "CALC-UA": "Coil E-101 heat-transfer coefficient (UA)",
    "CALC-APPROACH": "Steam-to-process temperature approach",
    "CALC-MBAL": "Mass balance residual",
    "ERR_TIC-101": "Temperature loop tracking error (PV - SP)",
    "ERR_LIC-101": "Level loop tracking error (PV - SP)",
    "ERR_FIC-201": "Steam flow loop tracking error (PV - SP)",
    "RES_DUTY": "Steam vs process duty residual",
    "UA_ROLLING_DIFF": "Coil fouling indicator (UA minus 24 h mean)",
    "VALVE_DISC_TV101": "Steam valve stiction (position - command)",
    "VALVE_DISC_LV101": "Level valve stiction (position - command)",
    "DIFF_TI_201_202": "Steam trap subcooling (TI-201 - TI-202)",
    "DIFF_TANK_IN_OUT": "Tank vs feed temperature rise",
    "DIFF_CORE_EXIT_TEMP": "Tank core vs outlet temperature",
}
for _t in ROLL_STD_TAGS:
    FEATURE_DESCRIPTIONS[f"{_t}_ROLL_STD"] = f"15-min volatility of {_t}"


def rolling_mean_1d(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.empty(n, dtype=np.float64)
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    if n >= window:
        out[window - 1 :] = (cumsum[window:] - cumsum[:-window]) / window
        counts = np.arange(1, window, dtype=np.float64)
        out[: window - 1] = cumsum[1:window] / counts
    else:  # shorter than one window: expanding mean
        out[:] = cumsum[1:] / np.arange(1, n + 1, dtype=np.float64)
    return out


def rolling_std_1d(arr: np.ndarray, window: int) -> np.ndarray:
    mean1 = rolling_mean_1d(arr, window)
    mean2 = rolling_mean_1d(arr * arr, window)
    out = np.sqrt(np.maximum(0.0, mean2 - mean1 * mean1))
    out[0] = 0.0
    return out


def impute(X_raw: np.ndarray, train_medians: np.ndarray) -> np.ndarray:
    X = X_raw.copy()
    r, c = np.where(np.isnan(X))
    X[r, c] = train_medians[c]
    return X


def engineer_features(X_imp: np.ndarray, tag_names: list[str]) -> tuple[np.ndarray, list[str]]:
    idx = {name: i for i, name in enumerate(tag_names)}

    def col(name: str) -> np.ndarray:
        return X_imp[:, idx[name]]

    roll_std_cols = [rolling_std_1d(col(t), STD_WINDOW) for t in ROLL_STD_TAGS]
    eng_cols = [
        col("TIC-101.PV") - col("TIC-101.SP"),
        col("LIC-101.PV") - col("LIC-101.SP"),
        col("FIC-201.PV") - col("FIC-201.SP"),
        col("CALC-DUTY") - col("CALC-DUTY-PROC"),
        col("CALC-UA") - rolling_mean_1d(col("CALC-UA"), UA_WINDOW),
        col("TV-101.POS") - col("FIC-201.OP"),
        col("LV-101.POS") - col("LIC-101.OP"),
        col("TI-201") - col("TI-202"),
        col("TIC-101.PV") - col("TI-100"),
        col("TIC-101.PV") - col("TI-102"),
        *roll_std_cols,
    ]
    eng_names = [
        "ERR_TIC-101",
        "ERR_LIC-101",
        "ERR_FIC-201",
        "RES_DUTY",
        "UA_ROLLING_DIFF",
        "VALVE_DISC_TV101",
        "VALVE_DISC_LV101",
        "DIFF_TI_201_202",
        "DIFF_TANK_IN_OUT",
        "DIFF_CORE_EXIT_TEMP",
        *[f"{t}_ROLL_STD" for t in ROLL_STD_TAGS],
    ]
    X_all = np.column_stack([X_imp, *eng_cols])
    all_names = tag_names + eng_names
    keep = [
        i
        for i, n in enumerate(all_names)
        if not (n.endswith(DROPPED_SUFFIXES) or n.startswith(DROPPED_PREFIXES) or n in DROPPED_EXACT)
    ]
    return X_all[:, keep], [all_names[i] for i in keep]
