import matplotlib
matplotlib.use('Agg')

from load_data import load_data
from threshold_markers import mark_thresholds
from plotting import plot_subject_data

def main():
    data, subjects = load_data('data/test_measure.csv', 'data/subject-info.csv')
    data = mark_thresholds(data, subjects)
    plot_subject_data(data)

if __name__ == '__main__':
    main()


# Replace missing beats with the mean of the available beats?
# Parse by start and end of the exercise test
# Parse RR-intervals by thresholds 
