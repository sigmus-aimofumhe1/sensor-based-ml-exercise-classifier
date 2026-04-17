# Barbell Exercise Classification 🏋️‍♂️

![Status](https://img.shields.io/badge/status-completed-brightgreen)

## Overview
This project develops a machine learning system to classify barbell exercises—such as squats, bench press, and deadlifts—using time-series sensor data from wearable devices.

In addition to classification, the project includes a **rep counting algorithm** that detects repetitions based on motion signal patterns, demonstrating a practical fitness tracking application.

---

## Dataset
The dataset consists of time-series motion sensor data, including:

- Accelerometer readings  
- Gyroscope readings  
- Timestamped signals  
- Exercise labels (e.g., squat, bench press, deadlift)  

---

## Methodology

### Data Preprocessing
- Data cleaning and noise reduction  
- Resampling time-series signals  
- Outlier detection and removal  

### Feature Engineering
- Statistical features (mean, standard deviation)  
- Sliding window segmentation  
- Temporal pattern extraction  

### Model Training
The following models were implemented and evaluated:
- Logistic Regression  
- Support Vector Machine (SVM)  
- Decision Tree  
- Random Forest  

### Evaluation
- Accuracy score  
- Mean Absolute Error (MAE)  
- Confusion matrix  

---

## Results ✅

The model achieves high performance in classifying barbell exercises from sensor data.

- **Best Model:** Random Forest  
- **Accuracy:** 97.458%  
- **MAE:** 1.02  
- **Key Features:** Mean, standard deviation, and sliding window features  

### Sample Predictions

| Actual Exercise | Predicted |
|----------------|----------|
| Squat          | Squat    |
| Bench Press    | Bench    |
| Deadlift       | Deadlift |

---

## Rep Counting

A rep counting system was implemented using peak detection in the sensor signal. This enables:

- Automatic repetition counting  
- Continuous exercise tracking  
- Extension toward real-time fitness monitoring systems  

---

## Project Structure

``` id="6pksvu"
sensor-based-ml-exercise-classifier/
│
├── data/               # Raw and processed sensor data
├── models/             # Trained machine learning models
├── notebooks/          # Exploratory analysis and experiments
├── reports/figures/    # Visualizations and plots
├── src/                # Source code (training, prediction, rep counting)
├── requirements.txt    # Python dependencies
├── environment.yml     # Environment configuration
└── README.md
