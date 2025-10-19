import numpy as np
import pandas as pd
from scipy.signal import welch


def sample_entropy(rr, m=2, r=0.2):
    rr = np.asarray(rr)
    N = len(rr)
    r *= np.std(rr)  
    def _phi(m):
        x = np.array([rr[i:i+m] for i in range(N - m + 1)])
        C = np.sum(np.max(np.abs(x[:, None] - x[None, :]), axis=2) <= r, axis=0) - 1
        return np.sum(C) / (N - m + 1)
    try:
        return -np.log(_phi(m + 1) / _phi(m))
    except (ZeroDivisionError, FloatingPointError):
        return np.nan
    

def compute_features(rr: np.ndarray, fs: float = 4.0, include_freq: bool = False) -> dict:
    rr = np.asarray(rr)
    rr_diff = np.diff(rr)

    sdnn = np.std(rr)
    rmssd = np.sqrt(np.mean(rr_diff ** 2))
    sampen = sample_entropy(rr, m=2, r=0.2)

    features = {
        "sdnn": sdnn,
        "rmssd": rmssd,
        "sampen": sampen,
    }

    if include_freq:
        f, pxx = welch(rr, fs=fs, nperseg=min(256, len(rr)))

        def band_power(freqs, power, band):
            mask = (freqs >= band[0]) & (freqs < band[1])
            return np.trapz(power[mask], freqs[mask]) if np.any(mask) else 0.0

        lf = band_power(f, pxx, (0.04, 0.15))
        hf = band_power(f, pxx, (0.15, 0.4))
        total_power = band_power(f, pxx, (0.0033, 0.4))
        lf_hf = lf / hf if hf > 0 else np.nan

        features.update({
            "lf": lf,
            "hf": hf,
            "lf_hf": lf_hf,
            "total_power": total_power
        })

    return features


def extract_hrv_features(classification_data: pd.DataFrame, include_freq=False) -> pd.DataFrame:
    features = []

    label_cols = [c for c in classification_data.columns if c != "RR"]

    for _, row in classification_data.iterrows():
        rr = row["RR"]
        feats = compute_features(rr, include_freq=include_freq)
        for label in label_cols:
            feats[label] = row[label]
        features.append(feats)

    return pd.DataFrame(features)

