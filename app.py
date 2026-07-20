import os
import joblib
import numpy as np
import gradio as gr

# Load trained model and scaler
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")


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
    thal,
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
        risk = "🔴 High Risk"
        advice = """
### Recommendation

- Consult a cardiologist.
- Maintain a heart-healthy diet.
- Exercise regularly.
- Avoid smoking and alcohol.
- Schedule ECG/ECHO if advised.
"""
    else:
        risk = "🟢 Low Risk"
        advice = """
### Recommendation

- Continue a healthy lifestyle.
- Exercise regularly.
- Eat a balanced diet.
- Get routine health checkups.
"""

    return f"""
# ❤️ Heart Disease Prediction

### Prediction
**{"Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"}**

### Risk Level
**{risk}**

### Probability
**{probability:.2%}**

{advice}
"""


demo = gr.Interface(
    fn=predict_heart_disease,
    inputs=[
        gr.Number(label="Age", value=50),
        gr.Radio([0, 1], label="Sex (0=Female, 1=Male)", value=1),
        gr.Dropdown([0, 1, 2, 3], label="Chest Pain Type", value=0),
        gr.Number(label="Resting Blood Pressure", value=120),
        gr.Number(label="Cholesterol", value=220),
        gr.Radio([0, 1], label="Fasting Blood Sugar (>120 mg/dl)", value=0),
        gr.Dropdown([0, 1, 2], label="Resting ECG", value=1),
        gr.Number(label="Maximum Heart Rate", value=150),
        gr.Radio([0, 1], label="Exercise Induced Angina", value=0),
        gr.Number(label="Oldpeak", value=1.0),
        gr.Dropdown([0, 1, 2], label="Slope", value=2),
        gr.Dropdown([0, 1, 2, 3], label="Major Vessels (CA)", value=0),
        gr.Dropdown([0, 1, 2, 3], label="Thalassemia", value=2),
    ],
    outputs=gr.Markdown(),
    title="❤️ AI-Powered Heart Disease Prediction",
    description="""
Enter the patient's clinical details and click **Submit**.

The prediction is generated using a trained Random Forest Machine Learning model.
""",
    theme=gr.themes.Soft(),
)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
    )
