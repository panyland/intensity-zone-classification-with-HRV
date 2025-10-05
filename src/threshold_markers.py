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