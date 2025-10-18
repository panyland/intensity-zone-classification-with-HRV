import pandas as pd
import matplotlib
matplotlib.use('Agg')

from preprocessing import mark_thresholds
from preprocessing import replace_missing_beats
from plotting import plot_subject_data


def load_data(measure_path, subject_path):
    data = pd.read_csv(measure_path)
    subjects = pd.read_csv(subject_path)
    return data, subjects 


def remove_pre_post_periods(df, power_column='power'):
    result_df = df.copy()
    result_df = result_df[result_df[power_column] != 0]
    
    return result_df.reset_index(drop=True)


def label_rr_intervals(df):
    result_df = df.copy()
    result_df['Sub_vt1'] = (result_df['power'] < result_df['P_vt1']).astype(int)
    result_df['Mid_vt'] = ((result_df['power'] >= result_df['P_vt1']) & (result_df['power'] < result_df['P_vt2'])).astype(int)
    result_df['Supra_vt2'] = (result_df['power'] > result_df['P_vt2']).astype(int)
    result_df['At_vt'] = ((result_df['vt1_marker'] == 1) | (result_df['vt2_marker'] == 1)).astype(int)

    result_df.loc[result_df['At_vt'] == 1, ['Sub_vt1', 'Mid_vt', 'Supra_vt2']] = 0

    return result_df.reset_index(drop=True)


def create_classification_dataset(df):
    result_df = df.copy()
    grouped = (
        result_df.groupby(['ID', 'power'])
        .agg({
            'RR': list,
            'Sub_vt1': 'max',
            'Mid_vt': 'max',
            'Supra_vt2': 'max',
            'At_vt': 'max'
        })
        .reset_index()
    )

    grouped = grouped[grouped['At_vt'] == 0]
    classification_df = grouped[['RR', 'Sub_vt1', 'Mid_vt', 'Supra_vt2']].reset_index(drop=True)
    return classification_df


def main():
    data, subjects = load_data('data/test_measure.csv', 'data/subject-info.csv')
    data = remove_pre_post_periods(data)
    
    data = mark_thresholds(data, subjects)
    data.to_csv('data/marked_test_measure.csv', index=False)

    data = replace_missing_beats(data)
    data.to_csv('data/cleaned_test_measure.csv', index=False)

    plot_subject_data(data)

    data = label_rr_intervals(data)
    data.to_csv('data/labeled_test_measure.csv', index=False) 

    classification_data = create_classification_dataset(data)
    
    n = 100 # Number of RR-intervals per sample
    classification_data = classification_data[classification_data['RR'].apply(len) >= n].copy()
    classification_data['RR'] = classification_data['RR'].apply(lambda x: x[-n:])
    classification_data = classification_data[classification_data['RR'].apply(lambda x: not any(pd.isna(v) for v in x))].copy()
    classification_data.to_csv('data/classification_dataset.csv', index=False)


if __name__ == '__main__':
    main()

