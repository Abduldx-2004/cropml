"""
train_model.py
---------------
Trains multiple ML models on the crop dataset, evaluates them, and saves
the final Random Forest model and preprocessing artifacts.

Usage:
    python backend/train_model.py

This script is well-commented for academic submission and beginner-friendly.
"""
import os
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, 'dataset', 'crop_data.csv')
MODEL_DIR = os.path.join(ROOT, 'model')
STATIC_DATA_DIR = os.path.join(ROOT, 'static', 'data')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DATA_DIR, exist_ok=True)


def load_and_preprocess(path=DATA_PATH):
    """Load CSV, handle missing values, encode labels, and scale features."""
    df = pd.read_csv(path)

    # Basic cleaning: drop rows with any missing values (dataset is small/synthetic)
    df = df.dropna()

    # Features and target
    X = df[['Temperature', 'Humidity', 'Rainfall', 'Soil_pH', 'Nitrogen', 'Phosphorus', 'Potassium']]
    y = df['Crop']

    # Label encode the target
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, le, scaler


def evaluate_models(X_train, X_test, y_train, y_test):
    """Train models, evaluate metrics, and return trained models and metrics dict."""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(probability=True, random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=500, random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='weighted', zero_division=0)
        rec = recall_score(y_test, preds, average='weighted', zero_division=0)
        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)

        results[name] = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1)
        }

    return models, results


def save_artifacts(final_model, label_encoder, scaler, metrics):
    """Save model, label encoder, scaler, and metrics JSON for frontend."""
    model_path = os.path.join(MODEL_DIR, 'random_forest_model.pkl')
    le_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    metrics_path = os.path.join(STATIC_DATA_DIR, 'metrics.json')

    joblib.dump(final_model, model_path)
    joblib.dump(label_encoder, le_path)
    joblib.dump(scaler, scaler_path)

    # Save a simplified metrics JSON used by Chart.js on frontend
    # We'll store accuracy per model for visualization
    acc_data = {name: round(m['accuracy'] * 100, 2) for name, m in metrics.items()}
    with open(metrics_path, 'w') as f:
        json.dump({'accuracy_percent': acc_data, 'full_metrics': metrics}, f, indent=2)

    print(f"Saved artifacts to {MODEL_DIR} and metrics to {metrics_path}")


def plot_accuracy(metrics, out_path=None):
    """Plot a bar chart of model accuracies and save to static/images."""
    import os
    labels = list(metrics.keys())
    accuracies = [metrics[k]['accuracy'] for k in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, accuracies, color=['#2e7d32', '#66bb6a', '#a5d6a7', '#c8e6c9', '#8bc34a', '#4caf50', '#9ccc65'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path)
        print(f"Saved accuracy chart to {out_path}")
    plt.close()


def main():
    X_train, X_test, y_train, y_test, le, scaler = load_and_preprocess()

    trained_models, metrics = evaluate_models(X_train, X_test, y_train, y_test)

    # Choose Random Forest as final model (per project requirement)
    final_model = trained_models['Random Forest']

    # Save artifacts
    save_artifacts(final_model, le, scaler, metrics)

    # Plot and save accuracy chart to static/images
    img_out = os.path.join(os.path.dirname(STATIC_DATA_DIR), 'images', 'accuracy_chart.png')
    plot_accuracy(metrics, img_out)

    # Also print metrics neatly
    print("\nModel evaluation metrics (accuracy, precision, recall, f1):")
    for name, m in metrics.items():
        print(f"{name}: acc={m['accuracy']:.3f}, prec={m['precision']:.3f}, rec={m['recall']:.3f}, f1={m['f1_score']:.3f}")


if __name__ == '__main__':
    main()
