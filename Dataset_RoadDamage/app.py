import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="AI-Based Road Damage Detection",
    layout="wide"
)

# -----------------------------------
# Load Model
# -----------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("road_damage_small.keras")

model = load_model()

# -----------------------------------
# Load Labels
# -----------------------------------
with open("label_mapping.json", "r") as f:
    class_indices = json.load(f)

class_names = {v: k for k, v in class_indices.items()}

# -----------------------------------
# Severity Function
# -----------------------------------
def get_severity(prediction_class, confidence):
    if prediction_class.lower() == "pothole":
        return "High"
    elif prediction_class.lower() == "crack":
        return "Medium"
    elif prediction_class.lower() == "manhole":
        return "Low"
    else:
        return "Unknown"

# -----------------------------------
# Recommendations
# -----------------------------------
def get_recommendation(severity):
    if severity == "High":
        return "Immediate maintenance recommended. High-risk road condition."
    elif severity == "Medium":
        return "Schedule repair inspection soon."
    elif severity == "Low":
        return "Monitor periodically for safety."
    return "No recommendation."

# ===================================
# SECTION 1 — HEADER
# ===================================
st.title("AI-Based Road Damage Detection System")
st.subheader("Smart City Infrastructure Monitoring using CNN")

st.markdown("---")

# ===================================
# SECTION 2 — ABOUT PROJECT
# ===================================
st.header("About the Project")

st.write("""
### Why Road Monitoring is Important
Road damage such as potholes and cracks can lead to:
- Vehicle damage
- Traffic accidents
- Increased maintenance costs

### Role of CNN in Computer Vision
Convolutional Neural Networks (CNNs) automatically extract image features to detect road surface damage accurately.

### Practical Industry Applications
- Smart city road monitoring
- Automated maintenance planning
- Highway safety systems
- Municipal infrastructure management
""")

st.markdown("---")

# ===================================
# SECTION 3 — UPLOAD AREA
# ===================================
st.header("Upload Road Image")

uploaded_file = st.file_uploader(
    "Upload Road Surface Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # ===================================
    # SECTION 4 — IMAGE PREVIEW
    # ===================================
    st.header("Uploaded Image Preview")

    img = Image.open(uploaded_file)

    st.image(
        img,
        caption="Uploaded Road Image",
        use_container_width=True
    )

    # Preprocessing
    img_resized = img.resize((128,128))
    img_array = image.img_to_array(img_resized)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction)*100
    predicted_class = class_names[predicted_index]

    severity = get_severity(predicted_class, confidence)
    recommendation = get_recommendation(severity)

    st.markdown("---")

    # ===================================
    # SECTION 5 — PREDICTION AREA
    # ===================================
    st.header("Prediction Results")

    st.success(f"Prediction: {predicted_class.upper()} Detected")
    st.info(f"Confidence: {confidence:.2f}%")
    st.warning(f"Severity: {severity}")

    st.markdown("---")

    # ===================================
    # SECTION 6 — VISUALIZATION AREA
    # ===================================
    st.header("Confidence Visualization")

    probs = prediction[0]

    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(
        [class_names[i] for i in range(len(probs))],
        probs * 100
    )

    ax.set_ylabel("Confidence (%)")
    ax.set_title("Class Confidence Graph")

    st.pyplot(fig)

    st.markdown("---")

    # ===================================
    # SECTION 7 — RECOMMENDATIONS
    # ===================================
    st.header("Maintenance Recommendations")

    st.write(recommendation)

    if severity == "High":
        st.error("Safety Warning: Immediate road repair required.")
    elif severity == "Medium":
        st.warning("Moderate risk detected. Repair recommended.")
    else:
        st.success("Low-risk condition.")
