import matplotlib
matplotlib.use('Agg')
import os
import zipfile

zip_path = "heartdisease.zip"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    print(zip_ref.namelist())

# extract zip file
with zipfile.ZipFile("heartdisease.zip", "r") as zip_ref:
    zip_ref.extractall("heart-disease_dataset")

print("Dataset Extracted Successfully.")

# load dataset
import pandas as pd

columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

df = pd.read_csv("heart-disease_dataset/Heart_disease_cleveland_new.csv")

df.head()

# converting target into binary
df["target"] = df["target"].apply(
    lambda x: 0 if x == 0 else 1
)

df["target"].value_counts()

# checking for null values
df.isnull().sum()

# replace missing values
import numpy as np

df.replace("?", np.nan, inplace=True)

df.isnull().sum()

# convert datatype as float
df = df.astype(float)

# filling missing data
df.fillna(df.median(), inplace=True)

df.isnull().sum()

# verify dataset
print(df.shape)
print(df.head())
print(df.info())
print(df.describe())

"""**Exploratory Data Analysis**"""

print("Rows n columns : ")
print(df.shape)

print(df.describe())

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

sns.countplot(
    x="target",
    data=df,
    palette="Set2"
)

plt.title("Heart Disease Distribution")

plt.show()

df["target"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    figsize=(6, 8)
)

plt.ylabel("")

plt.title("Target Distribution")

plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Matrix")

plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(
    df["age"],
    bins=15,
    kde=True,
    color="skyblue"
)

plt.title("Age Distribution")

plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(
    x="cp",
    hue="target",
    data=df,
    palette="viridis"
)

plt.title("Chest Pain Vs Heart Disease")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(
    x="sex",
    hue="target",
    data=df
)

plt.title("Gender Vs Heart Disease")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(
    x="target",
    y="chol",
    data=df,
    palette="Set2"
)

plt.title("Cholesterol Vs Heart Disease")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(
    x="age",
    y="thalach",
    hue="target",
    data=df,
    palette="coolwarm"
)

plt.title("Age vs Maximum Heart Rate")

plt.show()

sns.pairplot(
    df[["age", "chol", "thalach", "oldpeak", "target"]],
    hue="target",
    palette="Set2"
)

plt.show()

"""**Feature Engineering**"""

# feature selection
X = df.drop("target", axis=1)
y = df["target"]

print(X.head())
print(y.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training Data: ", X_train.shape)
print("Testing Data: ", X_test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Original Training Shape: ", X_train.shape)
print("Scaled Training Shape: ", X_train_scaled.shape)

scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)

print(scaled_df.head())

# model selection & training
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)
lr.fit(X_train_scaled, y_train)

# model prediction
prediction = lr.predict(X_test_scaled)

print(prediction[:10])

# model evaluation
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(
    y_test,
    prediction
)

print("Accuracy: ", accuracy)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_test,
    prediction
)

print(cm)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Disease", "Disease"],
    yticklabels=["No Disease", "Disease"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

from sklearn.metrics import classification_report

print(
    classification_report(
        y_test,
        prediction
    )
)

from sklearn.metrics import roc_curve, roc_auc_score

probablity = lr.predict_proba(
    X_test_scaled
)[:, 1]

fpr, tpr, threshold = roc_curve(y_test, probablity)
auc = roc_auc_score(
    y_test,
    probablity
)

plt.figure(figsize=(8, 5))
plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc:.3f}"
)
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("Roc Curve")
plt.legend()
plt.grid()
plt.show()

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    LogisticRegression(max_iter=1000),
    X_train_scaled,
    y_train,
    cv=5,
    scoring="accuracy"
)

print(cv_scores)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "K-NN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss")
}

results = []
for name, model in models.items():
    if name in ["Logistic Regression", "Support Vector Machine", "K-NN"]:
        model.fit(X_train_scaled, y_train)
        prediction = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    results.append({
        "Model": name,
        "Accuracy": accuracy
    })

    print(f"{name} done")

comparision = pd.DataFrame(results)
comparision = comparision.sort_values(by="Accuracy", ascending=False)

print(comparision)

from sklearn.metrics import precision_score, recall_score, f1_score

evaluation = []

for name, model in models.items():
    if name in ["Logistic Regression", "Support Vector Machine", "K-NN"]:
        prediction = model.predict(X_test_scaled)
    else:
        prediction = model.predict(X_test)

    evaluation.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, prediction),
        "Precision": precision_score(y_test, prediction),
        "Recall": recall_score(y_test, prediction),
        "F1-Score": f1_score(y_test, prediction)
    })

metrics_df = pd.DataFrame(evaluation)
print(metrics_df)

cv_results = []
for name, model_instance in models.items():
    if name in ["Logistic Regression", "Support Vector Machine", "K-NN"]:
        scores = cross_val_score(
            model_instance,
            X_train_scaled,
            y_train,
            cv=5
        )
    else:
        scores = cross_val_score(
            model_instance,
            X_train,
            y_train,
            cv=5
        )

    cv_results.append({
        "Model": name,
        "Accuracy": scores.mean(),
        "Standard Deviation": scores.std()
    })

cv_df = pd.DataFrame(cv_results)

cv_df.sort_values(
    by="Accuracy",
    ascending=False,
    inplace=True
)

print(cv_df)

for name, model_instance in models.items():
    if name in ["Logistic Regression", "Support Vector Machine", "K-NN"]:
        train_score = model_instance.score(
            X_train_scaled,
            y_train
        )
        test_score = model_instance.score(
            X_test_scaled, y_test
        )
    else:
        train_score = model_instance.score(
            X_train,
            y_train
        )
        test_score = model_instance.score(
            X_test, y_test
        )
    print(name)
    print("Training Accuracy: ", train_score)
    print("Testing Accuracy: ", test_score)

from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier(random_state=42)

parameter = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy"]
}

grid = GridSearchCV(
    estimator=rf,
    param_grid=parameter,
    cv=5,
    n_jobs=-1,
    scoring="accuracy"
)

grid.fit(X_train_scaled, y_train)
print("Best Parameter: ", grid.best_params_)
print("Best Score: ", grid.best_score_)

best_rf = grid.best_estimator_

print(best_rf)

prediction = best_rf.predict(X_test_scaled)

print("Accuracy")

print(accuracy_score(y_test, prediction))

print()

print(classification_report(
    y_test,
    prediction
))

from sklearn.model_selection import RandomizedSearchCV

params = {
    "n_estimators": [50, 100, 150, 200, 300],
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=params,
    n_iter=20,
    cv=5,
    random_state=42,
    n_jobs=-1
)

random_search.fit(
    X_train_scaled,
    y_train
)

print(random_search.best_params_)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance)

plt.figure(figsize=(10, 6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Importance")

plt.title("Feature Importance")

plt.gca().invert_yaxis()

plt.show()

import joblib

joblib.dump(
    best_rf,
    "heart_disease_model.pkl"
)

joblib.dump(
    scaler,
    "scaler.pkl"
)

print("Model Saved Successfully")

model = joblib.load(
    "heart_disease_model.pkl"
)

scaler = joblib.load(
    "scaler.pkl"
)


def predict_patient(patient_data):
    patient_scaled = scaler.transform([patient_data])

    prediction = model.predict(patient_scaled)[0]

    probability = model.predict_proba(patient_scaled)[0][1]

    print(f"Probability of Disease: {probability:.2%}")

    if prediction == 1:
        print("High Risk of Heart Disease")
    else:
        print("Low Risk of Heart Disease")


predict_patient([
    60, 1, 3, 145, 260,
    0, 1, 140, 1, 2.5,
    1, 1, 3
])

import sys
import subprocess

try:
    import gradio as gr
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio", "-q"])
    import gradio as gr


def predict_heart_disease(
    age,
    sex,
    cp,
    trestbps,
    chol,
    fbs,
    restecg,
    thalach,
    exang,
    oldpeak,
    slope,
    ca,
    thal
):

    patient = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    patient_scaled = scaler.transform(patient)
    prediction = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0][1]

    if prediction == 1:
        risk = "High"
        advice = """
Consult a cardiologist.
Maintain healthy diet.
Exercise regularly.
Avoid smoking.
Schedule ECG/ECHO.
"""
    else:
        risk = "Low"
        advice = """
Maintain healthy lifestyle.
Regular health check-up.
Balanced diet.
Continue physical activity.
"""

    return f"""
Prediction : {prediction},
Risk Level : {risk},
Probability : {probability:.2%},
Recommendation : {advice},
"""


demo = gr.Interface(
    fn=predict_heart_disease,
    inputs=[
        gr.Number(label="Age", value=50),
        gr.Radio([0, 1], label="Sex", value=1),
        gr.Dropdown([0, 1, 2, 3], label="Chest Pain Type", value=0),
        gr.Number(label="Resting Blood Pressure", value=120),
        gr.Number(label="Cholesterol", value=220),
        gr.Radio([0, 1], label="Fasting Blood Sugar", value=0),
        gr.Dropdown([0, 1, 2], label="Rest ECG", value=1),
        gr.Number(label="Maximum Heart Rate", value=150),
        gr.Radio([0, 1], label="Exercise Induced Angina", value=0),
        gr.Number(label="Oldpeak", value=1.0),
        gr.Dropdown([0, 1, 2], label="Slope", value=2),
        gr.Dropdown([0, 1, 2, 3], label="Major Vessels (CA)", value=0),
        gr.Dropdown([0, 1, 2, 3], label="Thalassemia", value=2)
    ],
    outputs=gr.Markdown(),
    title="AI-Powered Heart Disease Prediction System",
    description="Enter patient clinical details and click Submit to predict heart disease risk using the trained Random Forest model.",
    theme=gr.themes.Soft()
)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  
    print(f"Launching Gradio on port {port}...", flush=True)

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False
    )
