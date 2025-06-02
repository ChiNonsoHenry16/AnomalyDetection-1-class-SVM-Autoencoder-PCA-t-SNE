# Anomaly Detection using One class SVM, Autoencoder, Principal Component Analysis and PCA-t-SNE

This project implements three anomaly detection techniques—**Isolation Forest**, **One-Class SVM**, and **Autoencoder**—on the KDDCup'99 (NSL-KDD) intrusion detection dataset. The models are evaluated using confusion matrices, ROC and Precision-Recall curves. The Autoencoder model additionally includes threshold tuning and loss visualization.

## 🔍 Features

- 📊 Isolation Forest & One-Class SVM using scikit-learn
- 🌐 Dimensionality reduction with PCA and t-SNE for visualization
- 🤖 Autoencoder built with TensorFlow Keras
- ⚖️ SMOTE applied for balanced validation during Autoencoder training
- 🎯 Threshold optimization using F1-score
- 📈 Precision-Recall and ROC Curve plotting
- 📉 Loss curve for Autoencoder training
