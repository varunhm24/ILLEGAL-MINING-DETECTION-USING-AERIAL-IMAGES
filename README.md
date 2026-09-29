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
