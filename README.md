# Histologic Image Cancer Classification

## Overview
Built a deep learning model to classify histologic images for cancer diagnosis, addressing challenges such as class imbalance and high-resolution medical data.

## Problem
Manual classification of histologic images is time-consuming and prone to error, with a high risk of misclassification impacting patient outcomes.

## Approach
- Preprocessed high-resolution images (resizing, normalization, augmentation)  
- Addressed class imbalance through data grouping  
- Compared CNN architectures and Vision Transformer (ViT)  
- Evaluated models using accuracy, recall, and F1-score  

## Results
- Best Model: Vision Transformer (ViT)  
- Test Accuracy: **98.88%**  
- F1 Score: **98.91%**  
- Significantly outperformed CNN models, especially on minority class  

## Repository Structure
- `group_project/` – main deep learning project (image classification)  
- `sgd_optimization_from_scratch.ipynb` – implemented stochastic gradient descent from scratch  
- `perceptron_algorithms_from_scratch.py` – implemented perceptron and its variants from scratch  

> The primary focus of this repository is the histologic image cancer classification project.
