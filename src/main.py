import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# Preprocessing functions
from preprocessing import (
    label_rr_intervals,
    mark_thresholds,
    remove_pre_post_periods,
    replace_missing_beats,
)

# Plotting functions
from plotting import (
    plot_confusion_matrix,
    plot_importances,
    plot_patterns_by_class,
    plot_subject_data,
)


def load_data(measure_path, subject_path):
    data = pd.read_csv(measure_path)
    subjects = pd.read_csv(subject_path)
    return data, subjects 


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

    n = 100 # Number of RR-intervals per sample
    classification_df = classification_df[classification_df['RR'].apply(len) >= n].copy()
    classification_df['RR'] = classification_df['RR'].apply(lambda x: x[-n:])
    classification_df = classification_df[classification_df['RR'].apply(lambda x: not any(pd.isna(v) for v in x))].copy()

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
    classification_data.to_csv('data/classification_dataset.csv', index=False)

    label_counts = {
        'Sub_vt1': classification_data['Sub_vt1'].sum(),
        'Mid_vt': classification_data['Mid_vt'].sum(),
        'Supra_vt2': classification_data['Supra_vt2'].sum()
    }
    print("Classification dataset label counts:", label_counts)

    classification_data['VT_label_3class'] = classification_data[['Sub_vt1','Mid_vt','Supra_vt2']].idxmax(axis=1)
    classification_data['Supra_vt1'] = (classification_data['Mid_vt'] + classification_data['Supra_vt2']).clip(0,1)

    X = np.stack(classification_data['RR'].values)

    mode = '3class'  # '3class' or '2class'

    if mode == '3class':
        y = classification_data['VT_label_3class'].values
    elif mode == '2class':
        y = classification_data['Supra_vt1'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight='balanced',
        random_state=42
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)

    matrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(matrix, y_test)

    importances = rf.feature_importances_
    plot_importances(importances)

    plot_patterns_by_class(X, y)

    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
    print(f"\nMean CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")


if __name__ == '__main__':
    main()

# Extract better features from RR sequences (SDNN, RMSSD, pNN50, power spectral features, DFA...)