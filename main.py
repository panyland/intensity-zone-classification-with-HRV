import pandas as pd

data = pd.read_csv('data/test_measure.csv')
subjects = pd.read_csv('data/subject-info.csv')
print(subjects.head())