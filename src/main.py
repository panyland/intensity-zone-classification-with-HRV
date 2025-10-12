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


def main():
    data, subjects = load_data('data/test_measure.csv', 'data/subject-info.csv')
    
    data = mark_thresholds(data, subjects)
    data.to_csv('data/marked_test_measure.csv', index=False)

    data = replace_missing_beats(data)
    data.to_csv('data/cleaned_test_measure.csv', index=False)

    plot_subject_data(data)


if __name__ == '__main__':
    main()


# Replace missing beats with the mean of the available beats?
# Parse by start and end of the exercise test
# Parse RR-intervals by thresholds 
