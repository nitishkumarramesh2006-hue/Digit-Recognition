"""Recognise a handwritten digit from the dataset.

Usage: python predict.py          (random image)
       python predict.py 100      (image number 100)
"""
import sys
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
model = joblib.load("model.joblib")
idx = int(sys.argv[1]) if len(sys.argv) > 1 else int(__import__("random").randrange(len(digits.data)))
sample = digits.data[[idx]]
pred = model.predict(sample)[0]
print(f"Image #{idx}: predicted digit = {pred} (actual = {digits.target[idx]})")

plt.imshow(digits.images[idx], cmap="gray_r")
plt.title(f"Predicted: {pred}")
plt.axis("off")
plt.savefig("results/prediction.png", dpi=150, bbox_inches="tight")
