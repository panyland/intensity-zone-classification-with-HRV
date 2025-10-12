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


def replace_missing_beats(df, rr_column='RR', id_column='ID', upper_bound=2000, window_size=10):
    """
    Replaces missing beats (extra long RR-intervals) with the median of surrounding intervals.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing RR-interval data in long format
    rr_column : str, default 'RR'
        Name of the column containing RR-intervals
    id_column : str, default 'ID'
        Name of the column containing subject IDs
    upper_bound : int, default 2000
        Upper threshold in milliseconds. RR-intervals above this will be considered missing beats.
    window_size : int, default 10
        Number of surrounding intervals to use for median calculation (5 on each side)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with missing beats replaced by local median
    """
    
    
    result_df = df.copy()
    
    # Group by ID to process each subject separately
    for subject_id, subject_data in result_df.groupby(id_column):
        rr_values = subject_data[rr_column].values
        indices = subject_data.index
        
        cleaned_rr = rr_values.copy()
        
        for i in range(len(rr_values)):
            current_rr = rr_values[i]

            if pd.isna(current_rr) or current_rr is None:
                continue  
            
            # Check if current RR-interval exceeds upper bound
            if current_rr > upper_bound:
                
                start_idx = max(0, i - window_size//2)
                end_idx = min(len(rr_values), i + window_size//2 + 1)
                
                # Get the surrounding values (excluding the current outlier)
                surrounding_values = []
                for j in range(start_idx, end_idx):
                    if j != i and rr_values[j] <= upper_bound:  # Exclude current and other outliers
                        surrounding_values.append(rr_values[j])
                
                # If we have enough surrounding values, use their median
                if len(surrounding_values) >= window_size//2:  # At least half the window size
                    replacement_value = np.median(surrounding_values)
                    cleaned_rr[i] = replacement_value
                else:
                    # If not enough surrounding values, use global median for this subject
                    global_median = np.median([x for x in rr_values if x <= upper_bound])
                    cleaned_rr[i] = global_median                
        
        
        result_df.loc[indices, rr_column] = cleaned_rr
    
    return result_df