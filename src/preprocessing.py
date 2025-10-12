import pandas as pd
import numpy as np

def mark_thresholds(data, subjects):
    data = data.merge(subjects[['ID', 'P_vt1', 'P_vt2']], on='ID', how='left')
    data['vt1_marker'] = 0
    data['vt2_marker'] = 0

    for id_value, group in data.groupby('ID'):
        for vt, col in [('P_vt1', 'vt1_marker'), ('P_vt2', 'vt2_marker')]:
            p_vt = group[vt].iloc[0]
            power_values = group['power'].unique()
            power_values.sort()

            lower = power_values[power_values <= p_vt].max() if any(power_values <= p_vt) else None
            upper = power_values[power_values >= p_vt].min() if any(power_values >= p_vt) else None

            mask = (group['power'] == lower) | (group['power'] == upper)
            data.loc[mask & (data['ID'] == id_value), col] = 1

    return data 


def replace_missing_beats(df, rr_column='RR', id_column='ID', median_multiplier=1.2, window_size=10):
    df = df.copy()
    result_frames = []

    for subject_id, group in df.groupby(id_column):
        rr_values = group[rr_column].to_numpy(dtype=float).copy()
        n = len(rr_values)
        
        for i in range(n):
            start = max(0, i - window_size)
            end = min(n, i + window_size + 1)
            
            # Exclude the current value from the window
            window = np.concatenate((rr_values[start:i], rr_values[i+1:end]))
            
            # Remove NaNs from the window
            window = window[~np.isnan(window)]
            
            # Skip if no valid values left
            if len(window) == 0:
                continue

            median_val = np.median(window)

            # Skip if median is NaN (shouldn't happen after nan removal)
            if np.isnan(median_val):
                continue

            # Replace if abs difference is larger than threshold
            if abs(rr_values[i] - median_val) > median_val * (median_multiplier - 1):
                old_val = rr_values[i]
                rr_values[i] = median_val
                print(f"Replaced outlier RR value {old_val} at index {i} for ID {subject_id} with median {median_val}")

        group[rr_column] = rr_values
        result_frames.append(group)

    # Combine all processed groups
    cleaned_df = pd.concat(result_frames, ignore_index=True)

    return cleaned_df