# Principal Component Analysis (PCA) From Scratch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/library-NumPy-blueviolet.svg)](https://numpy.org/)

## 📌 Project Overview
This repository contains a modular, from-scratch implementation of **Principal Component Analysis (PCA)**. While high-level libraries like `scikit-learn` offer these tools out-of-the-box, this project focuses on the underlying linear algebra and eigenvalue decomposition that makes dimensionality reduction possible.

As a real-world application, the implementation is used to perform **Facial Recognition** on the Labeled Faces in the Wild (LFW) dataset, reducing thousands of pixels into a compact "latent space" for classification.

---

## 🚀 Key Features
* **Manual Implementation:** Built using only `NumPy` for the core algorithm (Mean-centering, Covariance, Eigen-decomposition, Projection).
* **Dimensionality Reduction:** Compresses high-dimensional image data while retaining maximum variance.
* **Machine Learning Pipeline:** Integrates PCA with a **Support Vector Classifier (SVC)** to demonstrate improved model performance and denoising.
* **Visualization:** Scripts to visualize "Eigenfaces" and the cumulative explained variance (Scree Plot).

---

## 🛠️ The Mathematics Behind It
The implementation follows a 6-step linear algebra pipeline to transform raw data into principal components:

1.  **Mean Centering:** Shifting the dataset so the mean of each feature is $0$.
2.  **Covariance Matrix Calculation:** Computing $\frac{1}{n-1} X^T X$ to find redundancies between features.
3.  **Eigenvalue Decomposition:** Solving $Av = \lambda v$ to find the "natural axes" of the data.
4.  **Component Sorting:** Ranking eigenvalues by magnitude to identify which directions hold the most information.
5.  **Feature Selection:** Choosing the top $k$ eigenvectors to form a projection matrix.
6.  **Projection:** Dotting the original data with the subset of eigenvectors to enter the reduced subspace.

---

## 💻 Installation & Usage

### Prerequisites
Ensure you have Python installed, then run:
```bash
pip install numpy matplotlib scikit-learn
