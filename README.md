# HRV-Based Intensity Classification Using Machine Learning

## Overview
This project explores the use of heart rate variability (HRV) to classify exercise intensity zones. The primary goal is to determine whether a short HRV measurement (1-3 minutes) can reliably indicate which intensity zone the subject is in — focusing initially on identifying the **second ventilatory threshold (VT2)**.

## Objectives
- Classify intensity zones using HRV (time or frequency domain features).
- Determine the **minimum time window** required for reliable classification.
- Use **support vector machines (SVMs)** for classification based on selected features.
- Try other models, like random forests or NNs, if necessary.

## Methodology
- Begin with VT2 classification using selected HRV features and SVM.
- Use data from **incremental exercise tests** where VT2 is known to occur between two workload steps.
- Ideally follow-up tests with **finer workload increments** between those steps could improve resolution.
(-> Repeat the process to refine the precision of VT2 detection.)

## Status
Data includes:
- **Power output**
- **RR-intervals**
- **VO₂ measurements**
- **Ventilatory thresholds**

These were collected from **graded exercise tests on a cycle ergometer** involving **18 subjects**, sourced from [this dataset](https://physionet.org/content/actes-cycloergometer-exercise/1.0.0/).

Model development and **time window optimization** for classification are currently in progress.

## Future Work
- Extend classification to other lactate thresholds.
- Validate model across different populations and test protocols.
