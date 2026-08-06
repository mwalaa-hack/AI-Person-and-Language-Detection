import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

RANDOM_STATE = 42

# 1. Load Dataset — train and test were already split (and only train was augmented)
# during preprocessing, so we load them separately here instead of splitting one file.
train_df = pd.read_csv("train_features.csv")
test_df = pd.read_csv("test_features.csv")

X_train = train_df.drop(columns=["Person", "Language", "Augmentation"])
y_train = train_df["Person"]

X_test = test_df.drop(columns=["Person", "Language", "Augmentation"])
y_test = test_df["Person"]

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
print("Classes:", list(le.classes_))
print("Train Feature matrix:", X_train.shape)
print("Test Feature matrix:", X_test.shape)

print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")


# 3. Pipeline: scaling + SVM
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(probability=True, random_state=RANDOM_STATE))
])

# Small dataset -> grid search wrapped in stratified CV instead of
# trusting a single validation split
param_grid = {
    "svm__C": [0.1, 1, 5, 10, 50, 100],
    "svm__gamma": ["scale", 0.001, 0.01, 0.1, 1],
    "svm__kernel": ["rbf"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

grid = GridSearchCV(
    pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=1
)
grid.fit(X_train, y_train)

print("\nBest params:", grid.best_params_)
print(f"Best CV accuracy: {grid.best_score_:.4f}")
best_model = grid.best_estimator_


# 4. Evaluate on held-out test set
print("Evaluating Best Model")
y_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"\nHeld-out test accuracy: {test_acc:.4f}\n")

print("Classification report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("Confusion matrix:")
print(pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    index=[f"true_{c}" for c in le.classes_],
    columns=[f"pred_{c}" for c in le.classes_]
))


# 5. Extra sanity check: 5-fold CV accuracy on the FULL dataset
#    (more reliable signal than a single split, given only 300 samples)
full_cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring="accuracy")
print(f"\nFull-dataset 5-fold CV accuracy: {full_cv_scores.mean():.4f} "
      f"(+/- {full_cv_scores.std():.4f})")


# 6. Save model + label encoder for later use / comparison
joblib.dump(best_model, "person_model_final_new.pkl")
joblib.dump(le, "person_label_encoder_new.pkl")

print("Model Saved Successfully")
print("person_model_final_new.pkl")
print("person_label_encoder_new.pkl")


#output
'''
Classes: ['EsraaM', 'MWalaa', 'MariamB']
Train Feature matrix: (976, 55)
Test Feature matrix: (61, 55)
Train: 976 samples | Test: 61 samples
Fitting 5 folds for each of 30 candidates, totalling 150 fits

Best params: {'svm__C': 5, 'svm__gamma': 'scale', 'svm__kernel': 'rbf'}
Best CV accuracy: 0.9979

Evaluating Best Model

Held-out test accuracy: 1.0000

Classification report:
              precision    recall  f1-score   support

      EsraaM       1.00      1.00      1.00        20
      MWalaa       1.00      1.00      1.00        21
     MariamB       1.00      1.00      1.00        20

    accuracy                           1.00        61
   macro avg       1.00      1.00      1.00        61
weighted avg       1.00      1.00      1.00        61

Confusion matrix:
              pred_EsraaM  pred_MWalaa  pred_MariamB
true_EsraaM            20            0             0
true_MWalaa             0           21             0
true_MariamB            0            0            20

Full-dataset 5-fold CV accuracy: 0.9979

Model Saved Successfully
person_model_final_new.pkl
person_label_encoder_new.pkl
'''