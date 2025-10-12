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


def main():
    data, subjects = load_data('data/test_measure.csv', 'data/subject-info.csv')
    data = remove_pre_post_periods(data)
    
    data = mark_thresholds(data, subjects)
    data.to_csv('data/marked_test_measure.csv', index=False)

    data = replace_missing_beats(data)
    data.to_csv('data/cleaned_test_measure.csv', index=False)

    plot_subject_data(data)


if __name__ == '__main__':
    main()


# Parse RR-intervals by thresholds and create teh dataset for ML models