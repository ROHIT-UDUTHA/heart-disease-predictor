import joblib
import gradio as gr
import numpy as np

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
        risk = "🚨 High Risk"
        recommendations = """
- 🩺 **Consult a Cardiologist immediately** for comprehensive evaluation.
- 🥗 **Maintain a Heart-Healthy Diet**: Low sodium, reduced saturated fats, high fiber.
- 🏃‍♂️ **Exercise Precautions**: Consult physician before physical exertion.
- 🚭 **Lifestyle**: Avoid smoking and limit alcohol intake.
- 📉 **Diagnostics**: Schedule ECG, Echocardiogram, and stress test.
"""
    else:
        risk = "✅ Low Risk"
        recommendations = """
- 💪 **Maintain Healthy Lifestyle**: Keep up regular physical activity and balanced nutrition.
- 🩺 **Routine Checkups**: Schedule annual health screenings.
- 🥦 **Balanced Diet**: Emphasize whole grains, lean proteins, vegetables, and fruit.
- 🧘‍♂️ **Wellness**: Practice stress management and ensure quality sleep.
"""

    result_md = f"""
### 📊 Prediction Result Summary

| Indicator | Result |
| :--- | :--- |
| **Risk Assessment** | **{risk}** |
| **Disease Probability** | **{probability:.2%}** |
| **Classification** | **{'Heart Disease Detected' if prediction == 1 else 'No Disease Detected'}** |

---

### 📋 Recommended Actions
{recommendations}
"""
    return result_md


demo = gr.Interface(
    fn=predict_heart_disease,
    inputs=[
        gr.Number(label="Age (Years)", value=50),
        gr.Radio(choices=[0, 1], label="Sex (0: Female, 1: Male)", value=1),
        gr.Dropdown(choices=[0, 1, 2, 3], label="Chest Pain Type (0: Typical Angina, 1: Atypical, 2: Non-anginal, 3: Asymptomatic)", value=0),
        gr.Number(label="Resting Blood Pressure (mm Hg)", value=120),
        gr.Number(label="Serum Cholesterol (mg/dl)", value=220),
        gr.Radio(choices=[0, 1], label="Fasting Blood Sugar > 120 mg/dl (0: No, 1: Yes)", value=0),
        gr.Dropdown(choices=[0, 1, 2], label="Resting ECG Results (0: Normal, 1: ST-T Abnormality, 2: LV Hypertrophy)", value=1),
        gr.Number(label="Maximum Heart Rate Achieved (Thalach)", value=150),
        gr.Radio(choices=[0, 1], label="Exercise Induced Angina (0: No, 1: Yes)", value=0),
        gr.Number(label="ST Depression Induced by Exercise (Oldpeak)", value=1.0),
        gr.Dropdown(choices=[0, 1, 2], label="Slope of Peak Exercise ST Segment (0: Upsloping, 1: Flat, 2: Downsloping)", value=2),
        gr.Dropdown(choices=[0, 1, 2, 3], label="Major Vessels Colored by Fluoroscopy (CA)", value=0),
        gr.Dropdown(choices=[0, 1, 2, 3], label="Thalassemia (0: Normal, 1: Fixed Defect, 2: Reversable Defect, 3: Other)", value=2)
    ],
    outputs=gr.Markdown(),
    title="❤️ AI-Powered Heart Disease Risk Assessment System",
    description="Enter clinical parameters to predict heart disease risk using the trained machine learning model."
)

if __name__ == "__main__":
    print("Launching Gradio App...", flush=True)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)
