import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

RANDOM_STATE = 42

# 1. Load data
df = pd.read_csv("preprocessed_features_2.csv")

feature_cols = [c for c in df.columns if c not in ("Person", "Language")]
X = df.drop(columns=["Person", "Language"]).values
y = df["Person"].values

le = LabelEncoder()
y = le.fit_transform(y)
print("Classes:", list(le.classes_))
print("Feature matrix:", X.shape)


# 2. Train / test split (stratified so each person is balanced)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
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
full_cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring="accuracy")
print(f"\nFull-dataset 5-fold CV accuracy: {full_cv_scores.mean():.4f} "
      f"(+/- {full_cv_scores.std():.4f})")


# 6. Save model + label encoder for later use / comparison
joblib.dump(best_model, "person_model.joblib")
joblib.dump(le, "person_label_encoder.joblib")
print("\nSaved: person_model.joblib, person_label_encoder.joblib")