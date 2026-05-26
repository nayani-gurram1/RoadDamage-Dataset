import streamlit as st
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt
import tflite_runtime.interpreter as tflite

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="AI-Based Road Damage Detection",
    layout="wide"
)

# -----------------------------------
# Load TFLite Model
# -----------------------------------
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(
        model_path="road_damage_model.tflite"
    )
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# -----------------------------------
# Load Labels
# -----------------------------------
with open("label_mapping.json", "r") as f:
    class_indices = json.load(f)

class_names = {v: k for k, v in class_indices.items()}

# -----------------------------------
# Severity Function
# -----------------------------------
def get_severity(prediction_class):
    if prediction_class.lower() == "pothole":
        return "High"
    elif prediction_class.lower() == "crack":
        return "Medium"
    elif prediction_class.lower() == "manhole":
        return "Low"
    return "Unknown"

# -----------------------------------
# Recommendation Function
# -----------------------------------
def get_recommendation(severity):
    if severity == "High":
        return "Immediate maintenance recommended. High-risk road condition."
    elif severity == "Medium":
        return "Schedule repair inspection soon."
    return "Monitor periodically for safety."

# -----------------------------------
# Header
# -----------------------------------
st.title("AI-Based Road Damage Detection System")
st.subheader("Smart City Infrastructure Monitoring using CNN")

st.markdown("---")

# -----------------------------------
# About
# -----------------------------------
st.header("About the Project")
st.write("""
Road monitoring helps prevent accidents and infrastructure damage.

CNN-based computer vision enables automatic detection of:
- Potholes
- Cracks
- Manholes

Applications:
- Smart city monitoring
- Highway inspection
- Maintenance prioritization
""")

st.markdown("---")

# -----------------------------------
# Upload
# -----------------------------------
uploaded_file = st.file_uploader(
    "Upload Road Surface Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_resized = img.resize((128,128))
    img_array = np.array(img_resized, dtype=np.float32)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(
        input_details[0]['index'],
        img_array
    )

    interpreter.invoke()

    prediction = interpreter.get_tensor(
        output_details[0]['index']
    )

    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction)*100
    predicted_class = class_names[predicted_index]

    severity = get_severity(predicted_class)

    st.success(f"Prediction: {predicted_class.upper()} Detected")
    st.info(f"Confidence: {confidence:.2f}%")
    st.warning(f"Severity: {severity}")

    # Visualization
    probs = prediction[0]

    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(
        [class_names[i] for i in range(len(probs))],
        probs*100
    )
    ax.set_ylabel("Confidence (%)")
    ax.set_title("Class Confidence Graph")

    st.pyplot(fig)

    # Recommendation
    st.header("Maintenance Recommendation")
    st.write(get_recommendation(severity))
