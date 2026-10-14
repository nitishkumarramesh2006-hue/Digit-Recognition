"""Handwritten Digit Recognition: SVM classifier.

Steps: load dataset -> baseline SVM -> evaluate -> tune hyperparameters
-> evaluate tuned model -> save model and results.
"""
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)

SEED = 42

# 1. Dataset: 1797 images, 8x8 pixels, classes 0-9
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"Images: {len(X)}, Pixels per image: {X.shape[1]}, Train/Test: {len(X_train)}/{len(X_test)}")

# Preview of the dataset
fig, axes = plt.subplots(2, 8, figsize=(10, 3))
for ax, img, lab in zip(axes.ravel(), digits.images, digits.target):
    ax.imshow(img, cmap="gray_r")
    ax.set_title(lab)
    ax.axis("off")
plt.suptitle("Sample images from the dataset")
plt.savefig("results/dataset_samples.png", dpi=150, bbox_inches="tight")
plt.close()


def evaluate(model, label):
    pred = model.predict(X_test)
    m = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision (macro)": precision_score(y_test, pred, average="macro"),
        "Recall (macro)": recall_score(y_test, pred, average="macro"),
        "F1 (macro)": f1_score(y_test, pred, average="macro"),
    }
    print(f"\n--- {label} ---")
    for k, v in m.items():
        print(f"{k}: {v:.4f}")
    return {k: round(v, 4) for k, v in m.items()}


# 2. Model implementation (baseline)
baseline = make_pipeline(StandardScaler(), SVC(random_state=SEED)).fit(X_train, y_train)
base_metrics = evaluate(baseline, "Baseline SVM")

# 3. Hyperparameter tuning
grid = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": ["scale", 0.001, 0.01, 0.1],
    "svc__kernel": ["rbf", "poly"],
}
search = GridSearchCV(make_pipeline(StandardScaler(), SVC(random_state=SEED)),
                      grid, cv=5, scoring="accuracy", n_jobs=-1)
search.fit(X_train, y_train)
print("\nBest parameters:", search.best_params_)
print(f"Best CV accuracy: {search.best_score_:.4f}")

# 4. Performance evaluation of tuned model
best = search.best_estimator_
tuned_metrics = evaluate(best, "Tuned SVM")
pred = best.predict(X_test)
print("\n", classification_report(y_test, pred))

joblib.dump(best, "model.joblib")
with open("results/metrics.json", "w") as f:
    json.dump({"baseline": base_metrics, "tuned": tuned_metrics,
               "best_params": search.best_params_}, f, indent=2)

ConfusionMatrixDisplay(confusion_matrix(y_test, pred)).plot(cmap="Blues")
plt.title("Confusion Matrix (Tuned SVM)")
plt.savefig("results/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

# Sample predictions
fig, axes = plt.subplots(2, 8, figsize=(10, 3))
for ax, img, p, t in zip(axes.ravel(), X_test, pred, y_test):
    ax.imshow(img.reshape(8, 8), cmap="gray_r")
    ax.set_title(f"P:{p} A:{t}", color="green" if p == t else "red", fontsize=9)
    ax.axis("off")
plt.suptitle("Predictions (P) vs Actual (A)")
plt.savefig("results/sample_predictions.png", dpi=150, bbox_inches="tight")
