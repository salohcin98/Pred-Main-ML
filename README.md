# Pred-Main-ML
Predictive Maintenance in Manufacturing: A Comparative Analysis of K-Means Clustering and Random Forest 

## Overview

This repository explores the application of machine learning models, specifically K-Means clustering and Random Forest Classification, for predictive maintenance in manufacturing. The goal is to predict machine failures, optimize maintenance strategies, and reduce operational costs.

## Table of Contents

- [Background](#background)
- [Data](#data)
- [Models](#models)
- [Results](#results)
- [Implementation](#implementation)
- [Conclusion](#conclusion)

## Background

Maintenance is a crucial aspect of manufacturing, and predictive maintenance offers a proactive approach to reduce costs and enhance operational efficiency. This project compares the effectiveness of K-Means clustering and Random Forest Classification in predicting machine failures.

## Data

The dataset comprises 10,000 data points with features like air temperature, process temperature, rotational speed, torque, and tool wear duration. The 'machine_failure' label indicates the occurrence of failures caused by various failure modes.

## Models

- **K-Means Clustering:** Attempts to identify patterns in machinery data for timely maintenance. Utilizes the elbow method and silhouette scoring to determine optimal cluster numbers.

- **Random Forest Classification:** Constructs an ensemble of decision trees to predict machine failures. Employs hyperparameter tuning for improved precision.

## Results

- **K-Means Clustering:** Shows limitations in predicting machine failures, with a focus on precision. Challenges include imbalanced datasets and struggles with actual failure predictions.

- **Random Forest Classification:** Demonstrates high overall accuracy with strong precision in predicting machine failures. However, it tends to overpredict failures, impacting recall.

## Implementation

The project is implemented in Python 3.8 using PyCharm as the IDE. Key libraries include Pandas for data management, Scikit-Learn for machine learning, and Matplotlib for data visualization. The dataset is preprocessed, features are selected, and models are trained and evaluated.

## Conclusion

The research provides insights into the strengths and limitations of K-Means clustering and Random Forest Classification for predictive maintenance. While K-Means struggles with imbalanced datasets, Random Forest shows promise in accurate failure predictions, despite some overprediction. The choice between models should align with industry requirements and dataset characteristics.


The dataset comes from S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications," 2020 Third International Conference on Artificial Intelligence for Industries (AI4I), 2020, pp. 69-74, doi: 10.1109/AI4I49448.2020.00023.
---
