# Automated Medical Image Classification for Radiology

This repository contains code and resources for a deep learning project that performs automated medical image classification. The goal is to classify radiology images, such as chest X-rays, to detect conditions like pneumonia, lung cancer, and fibrosis using convolution neural networks (CNNs).

## Project Structure

- `src`: Contains the main Python script `main.py`.
- `data`: Placeholder for the dataset, including `train` and `test` folders.
- `models`: Stores the saved trained model.
- `results`: Stores output images, metrics, and Grad-CAM visualizations.
- `notebooks`: Contains Jupyter notebooks for exploratory data analysis (EDA).

## Features

- **Data Augmentation**: Applied techniques like rescaling, rotation, and zooming.
- **Transfer Learning**: Used pre-trained CNN architectures for fine-tuning.
- **Grad-CAM**: Visualizes areas the model focuses on for predictions.
- **Evaluation**: Provides training, validation, and test metrics.

## Getting Started

### Requirements

1. Clone this repository:
   ```bash
   git clone https://github.com/AliNikoo73/Automated-Medical-Image-Classification.git
   cd Automated-Medical-Image-Classification

# Contributing to Automated Medical Image Classification

Thank you for considering contributing to this project!

## How to Contribute

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add a new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

### Issues
If you encounter any issues, please report them in the Issues section of this repository.

## Example Usage

To create and train the model, modify the parameters in the `main.py` file as needed and run:

```bash
python src/main.py