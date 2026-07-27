import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import joblib


# Load data
df = pd.read_csv("preprocessed_features_2.csv")
X = df.drop(columns=["Person", "Language"])
y = df["Language"]


# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,stratify=y,random_state=42)


# Cross Validation
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

# Models
models = {}
# Logistic Regression
models["Logistic Regression"] = Pipeline([("scaler", StandardScaler()),("model", LogisticRegression(max_iter=3000))])
# K Nearest Neighbors
models["KNN"] = Pipeline([("scaler", StandardScaler()),("model", KNeighborsClassifier(n_neighbors=5,weights="distance",metric="minkowski",p=2))])
# Random Forest
models["Random Forest"] = RandomForestClassifier(n_estimators=300,random_state=42)
# Support Vector Machine
models["SVM (RBF)"] = Pipeline([("scaler", StandardScaler()),("model", SVC(kernel="rbf",C=10,gamma="scale",probability=True,random_state=42))])
# XGBoost
models["XGBoost"] = XGBClassifier(n_estimators=300,learning_rate=0.05,max_depth=5,subsample=0.8,colsample_bytree=0.8,eval_metric="mlogloss",random_state=42)



# Train & Evaluate
results = []
best_model = None
best_name = None
best_score = -1

print("Language Classification Results")
for name, model in models.items():

    print("\n")
    print(name)

    cv_scores = cross_val_score(model,X_train,y_train,cv=cv,scoring="accuracy")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    precision = precision_score(y_test,predictions,average="macro")
    recall = recall_score(y_test,predictions,average="macro")
    f1 = f1_score(y_test,predictions,average="macro")

    print(f"Cross Validation Accuracy : {cv_scores.mean()*100:.2f}%")
    print(f"Test Accuracy             : {acc*100:.2f}%")
    print(f"Precision                : {precision:.4f}")
    print(f"Recall                   : {recall:.4f}")
    print(f"F1 Score                 : {f1:.4f}")

    print("\nClassification Report\n")
    print(classification_report(y_test,predictions,target_names=label_encoder.classes_))

    print("Confusion Matrix\n")
    print(confusion_matrix(y_test, predictions))

    results.append({"Model": name,"CV Accuracy": cv_scores.mean(),"Accuracy": acc,"Precision": precision,"Recall": recall,"F1": f1,})


    # Select best model using CV accuracy first,
    # then F1 as tie breaker.
    score = cv_scores.mean() + f1 / 100

    if score > best_score:
        best_score = score
        best_model = model
        best_name = name

# Results Table

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(by="CV Accuracy",ascending=False)

print("\n")
print("Summary")

print(results_df)


# Save best model
joblib.dump(best_model, "language_model.pkl")
joblib.dump(label_encoder, "language_label_encoder.pkl")

print("\n")
print("BEST MODEL")
print(best_name)
print("\nSaved as language_model.pkl")



# Report

'''
Language Classification Results 1


Logistic Regression
Cross Validation Accuracy : 75.68%
Test Accuracy             : 75.61%
Precision                : 0.7792
Recall                   : 0.7523
F1 Score                 : 0.7532

Classification Report

              precision    recall  f1-score   support

      Arabic       0.77      0.91      0.83        11
     English       0.86      0.60      0.71        10
      French       0.88      0.70      0.78        10
      German       0.62      0.80      0.70        10

    accuracy                           0.76        41
   macro avg       0.78      0.75      0.75        41
weighted avg       0.78      0.76      0.76        41

Confusion Matrix

[[10  0  0  1]
 [ 2  6  0  2]
 [ 0  1  7  2]
 [ 1  0  1  8]]


KNN
Cross Validation Accuracy : 75.04%
Test Accuracy             : 70.73%
Precision                : 0.7537
Recall                   : 0.7000
F1 Score                 : 0.6870

Classification Report

              precision    recall  f1-score   support

      Arabic       0.69      1.00      0.81        11
     English       0.73      0.80      0.76        10
      French       1.00      0.40      0.57        10
      German       0.60      0.60      0.60        10

    accuracy                           0.71        41
   macro avg       0.75      0.70      0.69        41
weighted avg       0.75      0.71      0.69        41

Confusion Matrix

[[11  0  0  0]
 [ 2  8  0  0]
 [ 0  2  4  4]
 [ 3  1  0  6]]


Random Forest
Cross Validation Accuracy : 81.74%
Test Accuracy             : 82.93%
Precision                : 0.8840
Recall                   : 0.8250
F1 Score                 : 0.8351

Classification Report

              precision    recall  f1-score   support

      Arabic       0.65      1.00      0.79        11
     English       1.00      0.70      0.82        10
      French       1.00      0.80      0.89        10
      German       0.89      0.80      0.84        10

    accuracy                           0.83        41
   macro avg       0.88      0.82      0.84        41
weighted avg       0.88      0.83      0.83        41

Confusion Matrix

[[11  0  0  0]
 [ 3  7  0  0]
 [ 1  0  8  1]
 [ 2  0  0  8]]


SVM (RBF)
Cross Validation Accuracy : 80.49%
Test Accuracy             : 80.49%
Precision                : 0.8299
Recall                   : 0.8023
F1 Score                 : 0.8078

Classification Report

              precision    recall  f1-score   support

      Arabic       0.67      0.91      0.77        11
     English       0.78      0.70      0.74        10
      French       1.00      0.90      0.95        10
      German       0.88      0.70      0.78        10

    accuracy                           0.80        41
   macro avg       0.83      0.80      0.81        41
weighted avg       0.83      0.80      0.81        41

Confusion Matrix

[[10  0  0  1]
 [ 3  7  0  0]
 [ 0  1  9  0]
 [ 2  1  0  7]]


XGBoost
Cross Validation Accuracy : 82.31%
Test Accuracy             : 85.37%
Precision                : 0.8806
Recall                   : 0.8500
F1 Score                 : 0.8529

Classification Report

              precision    recall  f1-score   support

      Arabic       0.73      1.00      0.85        11
     English       0.89      0.80      0.84        10
      French       0.90      0.90      0.90        10
      German       1.00      0.70      0.82        10

    accuracy                           0.85        41
   macro avg       0.88      0.85      0.85        41
weighted avg       0.88      0.85      0.85        41

Confusion Matrix

[[11  0  0  0]
 [ 2  8  0  0]
 [ 0  1  9  0]
 [ 2  0  1  7]]


Summary
                 Model  CV Accuracy  Accuracy  Precision    Recall        F1
4              XGBoost     0.823106  0.853659   0.880556  0.850000  0.852947
2        Random Forest     0.817424  0.829268   0.883987  0.825000  0.835059
3            SVM (RBF)     0.804924  0.804878   0.829861  0.802273  0.807805
0  Logistic Regression     0.756818  0.756098   0.779190  0.752273  0.753161
1                  KNN     0.750379  0.707317   0.753693  0.700000  0.687037


BEST MODEL
XGBoost


'''

#######################################################

'''
Language Classification Results 2


Logistic Regression
Cross Validation Accuracy : 80.44%
Test Accuracy             : 90.24%
Precision                : 0.9018
Recall                   : 0.9000
F1 Score                 : 0.8998

Classification Report

              precision    recall  f1-score   support

      Arabic       1.00      1.00      1.00        11
     English       0.90      0.90      0.90        10
      French       0.82      0.90      0.86        10
      German       0.89      0.80      0.84        10

    accuracy                           0.90        41
   macro avg       0.90      0.90      0.90        41
weighted avg       0.90      0.90      0.90        41

Confusion Matrix

[[11  0  0  0]
 [ 0  9  1  0]
 [ 0  0  9  1]
 [ 0  1  1  8]]


KNN
Cross Validation Accuracy : 73.18%
Test Accuracy             : 75.61%
Precision                : 0.7583
Recall                   : 0.7500
F1 Score                 : 0.7213

Classification Report

              precision    recall  f1-score   support

      Arabic       0.92      1.00      0.96        11
     English       0.67      1.00      0.80        10
      French       0.70      0.70      0.70        10
      German       0.75      0.30      0.43        10

    accuracy                           0.76        41
   macro avg       0.76      0.75      0.72        41
weighted avg       0.76      0.76      0.73        41

Confusion Matrix

[[11  0  0  0]
 [ 0 10  0  0]
 [ 1  1  7  1]
 [ 0  4  3  3]]


Random Forest
Cross Validation Accuracy : 80.51%
Test Accuracy             : 80.49%
Precision                : 0.8167
Recall                   : 0.8000
F1 Score                 : 0.7944

Classification Report

              precision    recall  f1-score   support

      Arabic       1.00      1.00      1.00        11
     English       0.82      0.90      0.86        10
      French       0.62      0.80      0.70        10
      German       0.83      0.50      0.62        10

    accuracy                           0.80        41
   macro avg       0.82      0.80      0.79        41
weighted avg       0.82      0.80      0.80        41

Confusion Matrix

[[11  0  0  0]
 [ 0  9  1  0]
 [ 0  1  8  1]
 [ 0  1  4  5]]


SVM (RBF)
Cross Validation Accuracy : 82.95%
Test Accuracy             : 90.24%
Precision                : 0.9129
Recall                   : 0.9000
F1 Score                 : 0.8974

Classification Report

              precision    recall  f1-score   support

      Arabic       1.00      1.00      1.00        11
     English       0.82      0.90      0.86        10
      French       0.83      1.00      0.91        10
      German       1.00      0.70      0.82        10

    accuracy                           0.90        41
   macro avg       0.91      0.90      0.90        41
weighted avg       0.92      0.90      0.90        41

Confusion Matrix

[[11  0  0  0]
 [ 0  9  1  0]
 [ 0  0 10  0]
 [ 0  2  1  7]]


XGBoost
Cross Validation Accuracy : 78.64%
Test Accuracy             : 87.80%
Precision                : 0.8874
Recall                   : 0.8750
F1 Score                 : 0.8721

Classification Report

              precision    recall  f1-score   support

      Arabic       1.00      1.00      1.00        11
     English       1.00      1.00      1.00        10
      French       0.69      0.90      0.78        10
      German       0.86      0.60      0.71        10

    accuracy                           0.88        41
   macro avg       0.89      0.88      0.87        41
weighted avg       0.89      0.88      0.88        41

Confusion Matrix

[[11  0  0  0]
 [ 0 10  0  0]
 [ 0  0  9  1]
 [ 0  0  4  6]]


Summary
                 Model  CV Accuracy  Accuracy  Precision  Recall        F1
3            SVM (RBF)     0.829545  0.902439   0.912879   0.900  0.897441
2        Random Forest     0.805114  0.804878   0.816725   0.800  0.794449
0  Logistic Regression     0.804356  0.902439   0.901768   0.900  0.899812
4              XGBoost     0.786364  0.878049   0.887363   0.875  0.872123
1                  KNN     0.731818  0.756098   0.758333   0.750  0.721273


BEST MODEL
SVM (RBF)

'''