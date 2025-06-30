# Handwritten Digit Recognition Project

## Overview

This project focuses on recognizing handwritten digits using machine learning techniques, specifically Decision Trees (D-Trees) and Support Vector Machines (SVM). It leverages feature extraction methods such as Histogram of Oriented Gradients (HOG), Sobel, and Laplacian filters to preprocess images and improve classification performance. The project includes hyperparameter tuning, K-fold cross-validation, and performance evaluation metrics like precision, recall, and F1-score to ensure robust model performance.

The document provided contains details on the implementation, challenges, and results of using these techniques to classify handwritten digits, addressing issues like scale sensitivity, high dimensionality, and class imbalance.This project focuses on recognizing handwritten digits using machine learning techniques, specifically Decision Trees (D-Trees) and Support Vector Machines (SVM). It leverages feature extraction methods such as Histogram of Oriented Gradients (HOG), Sobel, and Laplacian filters to preprocess images and improve classification performance. The project includes hyperparameter tuning, K-fold cross-validation, and performance evaluation metrics like precision, recall, and F1-score to ensure robust model performance.

The document provided contains details on the implementation, challenges, and results of using these techniques to classify handwritten digits, addressing issues like scale sensitivity, high dimensionality, and class imbalance.

## Features


- Feature Extraction:





- HOG and Sobel Filters: Extracts features sensitive to edge orientations and gradients, though they may struggle with scale and style variations in handwritten digits.



- Laplacian Filter: Highlights regions with significant intensity changes for edge detection, often used with Gaussian smoothing to reduce noise sensitivity.



- Machine Learning Models:





- Decision Trees (D-Trees): Optimized with hyperparameters like criterion='entropy', max_depth, min_samples_leaf, and min_samples_split to prevent overfitting and improve generalization.



- Support Vector Machines (SVM): Tuned with parameters like C and kernel functions to balance bias-variance and handle complex patterns.



- Hyperparameter Tuning:





- Best hyperparameters identified for D-Trees, e.g., {'criterion': 'entropy', 'max_depth': 15, 'min_samples_leaf': 5, 'min_samples_split': 5} for HOG + Laplacian features, achieving an accuracy of 0.86.



- SVM parameters like C and kernel choice optimized for effective pattern separation.



- Performance Evaluation:





- K-fold Cross-Validation: Used to tune parameters and prevent overfitting by limiting tree depth and sample splits.



- Metrics: Precision, recall, and F1-score calculated to evaluate model performance, particularly for imbalanced datasets.



- Challenges Addressed:





- Sensitivity to size and scale of handwritten digits.



- High-dimensional feature vectors leading to the "curse of dimensionality."



- Variability in handwriting styles and class imbalance in the dataset.



- Results:





- D-Tree models showed good performance for digits like 4 and 5 but struggled with digits like 3 due to scale and style variations.



- SVM models performed well overall but had issues distinguishing digits like 3, 7, and 9.



- HOG + Sobel features underperformed compared to other feature vectors.
