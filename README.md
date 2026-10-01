# 🛰️ Illegal Mining Detection Using Aerial Images

## Deep Learning-Based Aerial & Satellite Image Classification

[![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-ff4b4b?logo=streamlit)](https://streamlit.io/)
[![Computer Vision](https://img.shields.io/badge/Domain-Computer%20Vision-green)]()
[![Status](https://img.shields.io/badge/Project-Completed-success)]()

---

## 📌 Project Overview

Illegal mining can cause significant environmental damage, including deforestation, soil degradation, river pollution, and changes in land use.

This project presents a **deep learning-based computer vision system for detecting visual patterns associated with mining activity in aerial and satellite imagery**.

The system uses a **ResNet18 convolutional neural network (CNN)** to classify images into two visual categories:

- ⛏️ **Mining**
- 🌳 **Non-Mining**

A **Streamlit web application** is provided where users can upload an aerial/satellite image and receive the model's predicted class along with its confidence score.

> **Important:** The system performs visual classification based on the dataset's Mining and Non-Mining classes. It does not independently determine whether a mining operation is legally or illegally authorized. Legal status requires additional information such as permits, land boundaries, government records, and field verification.

---

# 🎯 Objectives

The main objectives of this project are:

1. Detect visual patterns associated with mining activity from aerial imagery.
2. Develop a deep learning image classification model.
3. Compare a custom CNN baseline with a pretrained ResNet18 model.
4. Evaluate the models using standard classification metrics.
5. Develop an interactive Streamlit application.
6. Provide a foundation for future satellite-based environmental monitoring systems.

---

# 🧠 Problem Statement

Manual monitoring of large geographical areas for mining activity can be time-consuming and difficult.

Satellite and aerial imagery provides a scalable source of information that can be analyzed using computer vision and deep learning.

The objective of this project is therefore to develop an automated image classification system capable of distinguishing between:

```text
Aerial / Satellite Image
          │
          ▼
     ResNet18 CNN
          │
          ▼
   ┌──────┴──────┐
   │             │
   ▼             ▼
Mining       Non-Mining

Key Features
🛰️ Aerial/satellite image classification
🧠 ResNet18 deep learning architecture
📊 Model performance evaluation
🔍 Confidence-based predictions
🖼️ Image upload interface
🌐 Streamlit web application
📁 Organized machine learning project structure
💾 Trained PyTorch model
📈 Baseline CNN comparison
⚠️ Responsible-use and legal-status disclaimer

🗂️ Dataset

The project uses the Illegal Mining Dataset available on Kaggle.

Dataset Source

Kaggle:

https://www.kaggle.com/datasets/lviacecliagomessilva/illegal-mining-dataset

Dataset Classes
Class	Description	Images
⛏️ Mining	Images belonging to the mining visual class	1,010
🌳 Non-Mining	Images belonging to the non-mining visual class	996
Total		2,006
Dataset Structure
Imagens_Dataset_Garimpo/
│
├── Imagens - Garimpo/
│   └── Mining Images
│
└── Imagens - Rios_Floresta/
    └── Non-Mining Images

The dataset is intentionally not included in this GitHub repository because of repository size and dataset management considerations.

🔬 Methodology

The project follows the following machine learning pipeline:

             Dataset
                │
                ▼
       Data Exploration
                │
                ▼
       Image Quality Check
                │
                ▼
       Stratified Data Split
                │
        ┌───────┴────────┐
        ▼                ▼
   Training Set      Validation Set
        │
        ▼
   CNN Training
        │
        ▼
   ResNet18 Training
        │
        ▼
     Evaluation
        │
        ▼
   Model Selection
        │
        ▼
   Prediction System
        │
        ▼
    Streamlit App
📊 Dataset Split

The dataset was divided using a stratified split to preserve the class distribution.

Dataset	Images
Training	1,404
Validation	301
Testing	301
Total	2,006
Training Distribution
Class	Images
Mining	707
Non-Mining	697
Validation Distribution
Class	Images
Mining	152
Non-Mining	149
Test Distribution
Class	Images
Mining	151
Non-Mining	150
🧪 Image Preprocessing

Images are resized to:

224 × 224 pixels

For ResNet18, ImageNet normalization is applied:

mean = [0.485, 0.456, 0.406]

std = [0.229, 0.224, 0.225]

The preprocessing pipeline is:

Original Image
      │
      ▼
Resize 224 × 224
      │
      ▼
Convert to Tensor
      │
      ▼
Normalize
      │
      ▼
ResNet18
🧠 Models
1. Custom CNN Baseline

A custom convolutional neural network was developed as a baseline model.

Architecture:

Input Image
     │
     ▼
Conv2D 3 → 32
     │
   ReLU
     │
 MaxPooling
     │
     ▼
Conv2D 32 → 64
     │
   ReLU
     │
 MaxPooling
     │
     ▼
Conv2D 64 → 128
     │
   ReLU
     │
 MaxPooling
     │
     ▼
Flatten
     │
     ▼
Fully Connected Layer
     │
     ▼
2 Output Classes

The baseline CNN achieved approximately 99.00% accuracy on the held-out test set.

🧠 2. ResNet18

The main model uses ResNet18, a convolutional neural network based on residual learning.

The pretrained ResNet18 classification head was modified for two classes:

model = models.resnet18(weights="DEFAULT")

num_features = model.fc.in_features

model.fc = nn.Linear(num_features, 2)
Training Configuration
Parameter	Value
Architecture	ResNet18
Input Size	224 × 224
Classes	2
Optimizer	Adam
Learning Rate	0.0001
Loss Function	Cross Entropy Loss
Epochs	10
Device	CPU
📈 Model Performance

The models were evaluated on a separate test set containing 301 images.

ResNet18 Test Results
Metric	Score
Accuracy	100.00%
Precision	100.00%
Recall	100.00%
F1-Score	100.00%
Confusion Matrix
                 Predicted
              Non-Mining  Mining

Actual
Non-Mining        150        0

Mining              0      151

The model correctly classified all 301 images in the held-out test set.

These results describe performance on this particular dataset and test split. They should not be interpreted as evidence of 100% accuracy on real-world satellite imagery.

🔍 Prediction Pipeline

The prediction system uses the trained ResNet18 model.

Upload Image
     │
     ▼
Image Preprocessing
     │
     ▼
Resize to 224×224
     │
     ▼
Normalization
     │
     ▼
ResNet18
     │
     ▼
Softmax Probabilities
     │
     ▼
Prediction + Confidence

Example output:

Prediction:
Mining

Confidence:
99.XX%

The confidence score represents the model's classification confidence and should not be interpreted as proof of illegal activity.

🌐 Streamlit Application

The project includes an interactive web application built using Streamlit.

Application Features
Upload aerial/satellite image
Preview uploaded image
Run model inference
Display predicted class
Display confidence percentage
Visual confidence indicator
Model information
Dataset information
Test performance
Responsible-use disclaimer
Application Flow
              Streamlit Web App
                     │
                     ▼
              Upload Image
                     │
                     ▼
              Preview Image
                     │
                     ▼
               Analyze Image
                     │
                     ▼
                ResNet18
                     │
              ┌──────┴──────┐
              ▼             ▼
           Mining       Non-Mining
              │             │
              └──────┬──────┘
                     ▼
                Confidence
🛠️ Technologies Used
Programming Language
Python
Machine Learning / Deep Learning
PyTorch
Torchvision
Scikit-learn
Computer Vision
Pillow
OpenCV
Data Analysis
NumPy
Pandas
Visualization
Matplotlib
Seaborn
Web Application
Streamlit
Development Tools
VS Code
Jupyter Notebook
Git
GitHub
📁 Project Structure
ILLEGAL-MINING-DETECTION-USING-AERIAL-IMAGES/
│
├── app/
│   └── app.py
│
├── models/
│   ├── baseline_cnn.pth
│   └── resnet18_mining.pth
│
├── notebooks/
│   └── 01_dataset_analysis.ipynb
│
├── src/
│   └── predict.py
│
├── .gitignore
│
└── README.md
Dataset

The dataset directory is intentionally excluded from GitHub:

Imagens_Dataset_Garimpo/

🔮 Future Scope

Several improvements can be explored in future versions.

1. Larger Dataset

Train using a larger and more geographically diverse dataset.

2. Satellite Data Integration

Integrate imagery from sources such as:

Sentinel-2
Landsat
Other remote sensing platforms
3. Geospatial Mapping

Add geographical coordinates to detected regions and visualize them on an interactive map.

4. Object Detection

Instead of classifying the entire image, use object detection models to identify mining-related regions.

Potential models:

YOLO
Faster R-CNN
RetinaNet
5. Image Segmentation

Use segmentation models to identify the exact area affected by mining.

Potential approaches:

U-Net
DeepLab
Mask R-CNN
6. Temporal Monitoring

Analyze satellite images over multiple dates to identify changes in land use.

Image at T1
     │
     ▼
Image at T2
     │
     ▼
Change Detection
     │
     ▼
Potential Mining Expansion
7. GIS Integration

Integrate the model with GIS platforms to provide:

Detection locations
Mining regions
Historical changes
Area estimates
Spatial analysis
8. Explainable AI

Add techniques such as Grad-CAM to show which regions of an image influenced the model's prediction.

⚠️ Limitations

This project is a research and academic prototype.

Important limitations include:

The model is trained on a specific dataset.
Dataset performance may not represent performance on unseen geographic regions.
Aerial image characteristics can vary significantly by location, season, resolution, and sensor.
The model performs visual classification rather than legal verification.
A prediction of the Mining class does not establish that mining is illegal.
Real-world deployment would require broader validation and additional geospatial and regulatory information.
🔐 Responsible Use

The system should be used as a decision-support and screening tool, not as an independent enforcement or legal-decision system.

A model prediction should be verified using appropriate sources such as:

Mining permits
Land ownership records
Geographic boundaries
Government databases
Field inspections
Additional satellite imagery
Human expert review
📚 Academic Context

This project demonstrates the application of:

Deep Learning
Computer Vision
Remote Sensing
Image Classification
Transfer Learning
Environmental Monitoring
Machine Learning Deployment

It provides an end-to-end workflow from dataset analysis and model training to deployment through a web application.

👨‍💻 Author

Varun H M

Engineering Student | AI/ML & Data Science Enthusiast

GitHub:

https://github.com/varunhm24

⭐ Project Highlights
🛰️ Aerial Image Analysis
🧠 ResNet18 Transfer Learning
📊 2,006 Image Dataset
⛏️ Mining vs Non-Mining Classification
🎯 100% Test Accuracy on 301-image Test Set
🌐 Streamlit Deployment
🐍 Python + PyTorch
📁 GitHub Portfolio Project
📜 Disclaimer

This project is developed for academic, research, and demonstration purposes.

The model identifies visual patterns corresponding to the dataset's Mining and Non-Mining classes. A model prediction alone does not establish that an activity is illegal. Any real-world decision should involve appropriate geographic, regulatory, legal, and human verification.
