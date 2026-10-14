# Handwritten Digit Recognition using SVM

A machine learning application that recognises handwritten digits (0-9) from 8x8 pixel images.

## Dataset
Optical Recognition of Handwritten Digits, 1797 images, 64 pixel features, 10 classes, built into scikit-learn.

## Project Workflow
1. **Dataset:** load and split 80/20 (stratified).
2. **Model implementation:** Support Vector Machine with feature scaling (baseline).
3. **Performance evaluation:** Accuracy, macro Precision, Recall, F1, confusion matrix.
4. **Hyperparameter tuning:** GridSearchCV (5-fold) over `C`, `gamma`, `kernel`.
5. **Before vs after tuning:** baseline and tuned metrics saved in `results/metrics.json`.

## Run
```bash
pip install -r requirements.txt
python train.py        # trains, tunes, evaluates, saves model.joblib and results/
python predict.py      # recognise one digit
```

## Results
See `results/metrics.json`, `results/confusion_matrix.png`, `results/sample_predictions.png`.
