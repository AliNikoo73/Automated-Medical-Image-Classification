# 🩺 **Lung Disease Classification Using Fine-Tuned Pre-Trained Models on X-ray Images**

---

## 📜 **Summary**

This project leverages the power of **deep learning** to classify **lung diseases** using X-ray images. The dataset consists of five classes:

- **Bacterial Pneumonia**
- **Coronavirus Disease**
- **Tuberculosis**
- **Viral Pneumonia**
- **Normal**

We use **pre-trained models** from the Keras library, such as **VGG16, ResNet152V2, and DenseNet201**, which are fine-tuned for optimal performance. The models undergo multiple trials, where **data augmentation** and preprocessing techniques are applied to improve generalization.

Key performance metrics such as **accuracy**, **loss**, and **confusion matrices** are generated, along with **Grad-CAM** heatmaps to interpret and visualize the model’s decisions.

💡 **Significance:** Early and accurate detection of lung diseases can greatly improve patient outcomes, especially in **resource-limited** settings where such technology can augment healthcare delivery.

---

## 🎯 **Objective**
> To accurately classify lung diseases using fine-tuned pre-trained models and interpret model decisions using visual explainability techniques like Grad-CAM.

---
### **Technical Skills** 

![Python](https://img.shields.io/badge/-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/-Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

- **Python** (TensorFlow, Keras, NumPy, Pandas)

![Deep Learning](https://img.shields.io/badge/-Deep%20Learning-FF6F00?style=for-the-badge&logo=deeplearning.ai&logoColor=white)
- **Deep Learning** (CNN architectures, Transfer Learning)

![Image Processing](https://img.shields.io/badge/-Image%20Processing-3498DB?style=for-the-badge&logo=opencv&logoColor=white)
- **Image Processing** (Data Augmentation, Normalization)

![Fine-Tuning](https://img.shields.io/badge/-Fine--Tuning-7D3C98?style=for-the-badge&logo=tensorflow&logoColor=white)
- **Model Fine-Tuning** (Pre-trained Models, Training Techniques)

![Grad-CAM](https://img.shields.io/badge/-Grad--CAM-FF4500?style=for-the-badge&logo=google&logoColor=white)
- **Grad-CAM** (Model Interpretability)

![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white)
- **Data Visualization** (matplotlib)

### **Soft Skills**
- 🔍 **Analytical Thinking**
- 🧠 **Problem-Solving**
- 🎯 **Attention to Detail**
- 📚 **Research & Adaptability**
---

## 📝 **Project Outputs**

### **Deliverables**
- 🏗 **Fine-Tuned Models** (e.g., VGG16, ResNet152V2, DenseNet201)
- 🧮 **Confusion Matrices** & **ROC Curves**
- 📊 **Model Performance Comparison**: Accuracy & Loss (Training, Validation, Testing)
- 🔥 **Grad-CAM Heatmaps** for Explainability
- 📈 **Bar Charts** Comparing Model Metrics

---

## 🔍 **Additional Details**

- **🗂 Dataset Source:** [Kaggle - Lung Disease Dataset (4 types)](https://www.kaggle.com/datasets/omkarmanohardalvi/lungs-disease-dataset-4-types)
- **🌐 Real-World Applicability:** Early and accurate detection of lung diseases through AI-based solutions.
- **💡 Challenges Overcome:**
  - Fine-tuning multiple **pre-trained models**.
  - Addressing **class imbalance** via **data augmentation** techniques.
- **🌍 Impact:** This project offers a scalable and automated solution for healthcare providers, especially in **underserved** areas where radiologists are scarce.

---

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
