import numpy as np
import pandas as pd
from scipy.signal import welch
from nolds import dfa # This fucking library causes headaches


def compute_features(rr: np.ndarray) -> dict:
    rr = np.asarray(rr)
    rr_diff = np.diff(rr)

    sdnn = np.std(rr)
    rmssd = np.sqrt(np.mean(rr_diff ** 2))
    
    try:
        alpha1 = dfa(rr, nvals=[4, 16])
        if isinstance(alpha1, (list, np.ndarray)):
            alpha1 = np.mean(alpha1)
    except Exception:
        alpha1 = np.nan
    
    return {
        "sdnn": sdnn,
        "rmssd": rmssd,
        "dfa_alpha1": alpha1,
    }


def extract_hrv_features(classification_data: pd.DataFrame) -> pd.DataFrame:
    features = []

    label_cols = [c for c in classification_data.columns if c != "RR"]

    for _, row in classification_data.iterrows():
        rr = row["RR"]
        feats = compute_features(rr)
        for label in label_cols:
            feats[label] = row[label]
        features.append(feats)

    return pd.DataFrame(features)