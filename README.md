# Barbell Exercise Classification 🏋️‍♂️

## Overview

This project builds a machine learning model to classify barbell exercises (such as squats, bench press, and deadlifts) using time-series sensor data.

---

## Dataset

* Accelerometer data
* Gyroscope data
* Timestamp
* Exercise labels

---

## Methodology

* Data preprocessing (cleaning, resampling, outliers)
* Feature engineering (mean, std, sliding windows)
* Model training (Logistic Regression, SVM, Decision Tree)
* Evaluation (accuracy, confusion matrix)

---

## Results
The model successfully classifies barbell exercises using sensor data.

- Best Model: Random Forest
- Accuracy: 99.458%
- Evaluation: MAE - 1.02
- Key Features: Mean, Standard Deviation, Sliding Window Features

### Sample Predictions
| Actual | Predicted |
|--------|----------|
| Squat  | Squat    |
| Bench  | Bench    |
| Deadlift | Deadlift |

### Rep Counting
A basic rep counting algorithm was implemented using signal peaks, allowing automatic counting of repetitions during exercise.

---

## Future Work

* Improve feature engineering
* Try more models
* Add rep counting

---

## Author
Eshiobomhe Sigmus Aimofumhe
