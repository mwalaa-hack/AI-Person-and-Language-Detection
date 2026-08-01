import pandas as pd
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
from sklearn.svm import SVC
import joblib

# Load Dataset — train and test were already split (and only train was augmented)
# during preprocessing, so we load them separately here instead of splitting one file.
train_df = pd.read_csv("train_features.csv")
test_df = pd.read_csv("test_features.csv")

X_train = train_df.drop(columns=["Person", "Language", "Augmentation"])
y_train = train_df["Language"]

X_test = test_df.drop(columns=["Person", "Language", "Augmentation"])
y_test = test_df["Language"]


# Encode Labels — fit only on train, then reuse the same encoder for test
# so label indices are consistent across both sets.
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)


# SVM Pipeline
pipeline = Pipeline([("scaler", StandardScaler()),("model", SVC(probability=True,random_state=42))])


# Hyperparameter Search
param_grid = {"model__kernel": ["rbf"],"model__C": [2, 3, 4, 5, 6, 7, 8, 10],"model__gamma": ["scale",0.02,0.01,0.005]}

cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

grid = GridSearchCV(estimator=pipeline,param_grid=param_grid,scoring="accuracy",cv=cv,n_jobs=-1,verbose=2,)

print("Searching for the best SVM parameters...\n")

grid.fit(X_train, y_train)


# Best Model
best_model = grid.best_estimator_
print("Best Parameters")
print(grid.best_params_)

print("\nBest Cross Validation Accuracy")
print(f"{grid.best_score_ * 100:.2f}%")

print("Evaluating Best Model")


# Prediction
predictions = best_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test,predictions,average="macro")
recall = recall_score(y_test,predictions,average="macro")
f1 = f1_score(y_test,predictions,average="macro")

print(f"\nTest Accuracy : {accuracy * 100:.2f}%")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1 Score      : {f1:.4f}")

print("\nClassification Report\n")

print(classification_report(y_test,predictions,target_names=label_encoder.classes_,))

print("Confusion Matrix\n")
print(confusion_matrix(y_test, predictions))


# Save Model

joblib.dump(best_model, "language_model_final_new.pkl")
joblib.dump(label_encoder, "language_label_encoder_new.pkl")

print("Model Saved Successfully")
print("language_model_final_new.pkl")
print("language_label_encoder_new.pkl")


#old
'''
Best Cross Validation Accuracy
83.56%
Evaluating Best Model

Test Accuracy : 90.24%
Precision     : 0.9129
Recall        : 0.9000
F1 Score      : 0.8974

Classification Report

              precision    recall  f1-score   support

      Arabic       1.00      1.00      1.00        11
     English       0.82      0.90      0.86        10
      French       0.83      1.00      0.91        10
      German       1.00      0.70      0.82        10

    accuracy                           0.90        41
   macro avg       0.91      0.90      0.90        41
weighted avg       0.92      0.90      0.90        41
'''


#new

''' 
Best Cross Validation Accuracy
92.62%
Evaluating Best Model

Test Accuracy : 93.44%
Precision     : 0.9351
Recall        : 0.9342
F1 Score      : 0.9344

Classification Report

              precision    recall  f1-score   support

      Arabic       0.92      0.97      0.94        62
     English       0.98      0.93      0.96        61
      French       0.92      0.92      0.92        59
      German       0.92      0.92      0.92        62

    accuracy                           0.93       244
   macro avg       0.94      0.93      0.93       244
weighted avg       0.94      0.93      0.93       244

Confusion Matrix

[[60  0  0  2]
 [ 0 57  2  2]
 [ 4  0 54  1]
 [ 1  1  3 57]]
'''