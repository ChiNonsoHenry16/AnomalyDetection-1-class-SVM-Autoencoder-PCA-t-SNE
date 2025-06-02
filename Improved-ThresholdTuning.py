import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE  # NEW
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers

# --- Load dataset ---
df = pd.read_csv("dataset/KDDTrain+.txt", header=None)
columns = [f'feature_{i}' for i in range(41)] + ['label', 'difficulty']
df.columns = columns
df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

# Encode categorical features
categorical_columns = [1, 2, 3]  # protocol_type, service, flag
for col in categorical_columns:
    df[f'feature_{col}'] = LabelEncoder().fit_transform(df[f'feature_{col}'])

X = df.drop(columns=['label', 'difficulty'])
y = df['label']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# --- SMOTE for balanced evaluation ---
X_balanced, y_balanced = SMOTE(random_state=42).fit_resample(X_scaled, y)

# --- Isolation Forest ---
iso_model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
iso_model.fit(X_scaled)
y_pred_iso = np.where(iso_model.predict(X_scaled) == -1, 1, 0)

# --- One-Class SVM ---
ocsvm_model = OneClassSVM(kernel='rbf', nu=0.05)
ocsvm_model.fit(X_scaled)
y_pred_ocsvm = np.where(ocsvm_model.predict(X_scaled) == -1, 1, 0)

# --- Autoencoder with threshold tuning ---
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
input_dim = X_train.shape[1]
encoding_dim = 14

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(1e-5))(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)
autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

history = autoencoder.fit(X_train, X_train,
                          epochs=10,
                          batch_size=256,
                          shuffle=True,
                          validation_data=(X_test, X_test),
                          verbose=0)

X_pred_full = autoencoder.predict(X_scaled)
mse_full = np.mean(np.power(X_scaled - X_pred_full, 2), axis=1)

X_val_pred = autoencoder.predict(X_test)
mse_val = np.mean(np.power(X_test - X_val_pred, 2), axis=1)

# Threshold tuning
precision, recall, thresholds = precision_recall_curve(y_test, mse_val)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_threshold = thresholds[np.argmax(f1_scores)]
y_pred_auto = (mse_full > best_threshold).astype(int)

# --- Evaluation Function ---
def evaluate_model(name, y_true, y_pred, scores=None):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    print(f"\n{name} Confusion Matrix:\n", cm)
    print(f"{name} Classification Report:\n", report)

    df_cm = pd.DataFrame(cm, index=['Normal', 'Anomaly'], columns=['Normal', 'Anomaly'])
    df_cm.to_csv(os.path.join(results_dir, f"{name.lower()}_confusion_matrix.csv"))
    plt.figure(figsize=(6, 4))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(os.path.join(results_dir, f"{name.lower()}_confusion_matrix.png"))
    plt.close()

    if scores is not None:
        # ROC
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.title(f'{name} ROC Curve')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend()
        plt.savefig(os.path.join(results_dir, f"{name.lower()}_roc_auc.png"))
        plt.close()

        # Precision-Recall
        prec, rec, _ = precision_recall_curve(y_true, scores)
        pr_auc = average_precision_score(y_true, scores)
        plt.figure()
        plt.plot(rec, prec, label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.title(f'{name} Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.savefig(os.path.join(results_dir, f"{name.lower()}_precision_recall.png"))
        plt.close()

# --- Evaluate All Models ---
evaluate_model("IsolationForest", y, y_pred_iso, scores=-iso_model.decision_function(X_scaled))
evaluate_model("OneClassSVM", y, y_pred_ocsvm, scores=-ocsvm_model.decision_function(X_scaled))
evaluate_model("Autoencoder", y, y_pred_auto, scores=mse_full)

# --- Autoencoder Loss Plot ---
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Autoencoder Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(results_dir, "autoencoder_loss.png"))
plt.close()

# --- PCA Visualization ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure()
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette=['blue', 'red'], alpha=0.5)
plt.title("PCA Visualization")
plt.savefig(os.path.join(results_dir, "pca_plot.png"))
plt.close()

# --- t-SNE Visualization ---
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=500)
X_tsne = tsne.fit_transform(X_scaled)
plt.figure()
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette=['green', 'orange'], alpha=0.5)
plt.title("t-SNE Visualization")
plt.savefig(os.path.join(results_dir, "tsne_plot.png"))
plt.close()
