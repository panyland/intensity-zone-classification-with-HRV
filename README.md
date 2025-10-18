# HRV-Based Intensity Zone Classification Using Machine Learning

## Overview
This project explores the use of heart rate variability (HRV) to classify exercise intensity zones. The primary goal is to determine whether a short RR-interval sequence (~1 min) can reliably indicate which intensity zone the subject is in.

## Objectives
- Classify intensity zones using HRV (time or frequency domain features).
- Determine the **minimum time window** required for reliable classification.
- Use **Random Forest** for classification based on selected features.
- Try other models, like NNs, if necessary.

## Methodology
- Use data from **incremental exercise tests** where VT1 and VT2 are known to occur between two workload steps.
- Ideally follow-up tests with **finer workload increments** between those steps could improve resolution.
(-> Repeat the process to refine the precision of VT detection.)

## Data
Includes:
- **Power output**
- **RR-intervals**
- **VO₂ measurements**
- **Ventilatory thresholds**

These were collected from **graded exercise tests on a cycle ergometer** involving **18 subjects**, sourced from [this dataset](https://physionet.org/content/actes-cycloergometer-exercise/1.0.0/).

Model development and **time window optimization** for classification are currently in progress.

## Future Work
- Extend classification to lactate thresholds.
- Validate model across different populations and test protocols.
