# Automated Medical Image Classification for Radiology

This repository contains code and resources for a deep learning project that performs automated medical image classification. The goal is to classify radiology images, such as chest X-rays, to detect conditions like pneumonia, lung cancer, and fibrosis using convolutional neural networks (CNNs).

## Project Structure

- `src`: Contains the main Python script `main.py`.
- `data`: Placeholder for the dataset, including `train` and `test` folders.
- `models`: Stores the saved trained model.
- `results`: Stores output images, metrics, and Grad-CAM visualizations.
- `notebooks`: Contains Jupyter notebooks for exploratory data analysis (EDA).

## Features

- **Data Augmentation**: Applied techniques like rescaling, rotation, and zooming.
- **Transfer Learning**: Used pretrained CNN architectures for fine-tuning.
- **Grad-CAM**: Visualizes areas the model focuses on for predictions.
- **Evaluation**: Provides training, validation, and test metrics.

## Getting Started

### Requirements

1. Clone this repository:
   ```bash
   git clone https://github.com/YourUsername/Automated-Medical-Image-Classification.git
   cd Automated-Medical-Image-Classification
