# Anomaly Detection using One class SVM, Autoencoder, Principal Component Analysis and t-SNE

This project implements three anomaly detection techniques—**Isolation Forest**, **One-Class SVM**, and **Autoencoder**—on the KDDCup'99 (NSL-KDD) intrusion detection dataset. The models are evaluated using confusion matrices, ROC and Precision-Recall curves. The Autoencoder model additionally includes threshold tuning and loss visualization.

## 🔍 Features

- 📊 Isolation Forest & One-Class SVM using scikit-learn
- 🌐 Dimensionality reduction with PCA and t-SNE for visualization
- 🤖 Autoencoder built with TensorFlow Keras
- ⚖️ SMOTE applied for balanced validation during Autoencoder training
- 🎯 Threshold optimization using F1-score
- 📈 Precision-Recall and ROC Curve plotting
- 📉 Loss curve for Autoencoder training

## 🛠 Requirements

- Python 3.7+
- TensorFlow
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn
- pandas
- numpy

Install dependencies:

```bash
pip install -r requirements.txt

```

▶️ Usage
1. Place the dataset file (KDDTrain+.txt) into the dataset/ folder.
2. Run the Python script:

```bash
python anomaly_detection.py
```

3. Output files (confusion matrices, plots) will be saved in the results/ folder.
   
autoencoder_loss.png: Loss curve of the Autoencoder training
autoencoder_roc_auc.png: ROC Curve for Autoencoder
autoencoder_precision_recall.png: PR Curve for Autoencoder
pca_plot.png: 2D PCA plot of the dataset
tsne_plot.png: 2D t-SNE plot of the dataset

## 📌 Notes

SMOTE is used only during validation to balance classes, not for training unsupervised models.
Threshold tuning is done for the Autoencoder using the F1 score from Precision-Recall curves.
