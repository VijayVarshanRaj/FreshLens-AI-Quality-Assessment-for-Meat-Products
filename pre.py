import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# ---------- CONFIG ----------
st.set_page_config(
    page_title="FreshLens AI: Quality Assessment for Meat Products",
    layout="centered"
)

MODEL_PATH = "model_v1_meat.h5"
IMAGE_SIZE = (128, 128)
CLASS_NAMES = ['Fresh', 'Half-Fresh', 'Spoiled']
CLASS_COLORS = ['green', 'orange', 'red']

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model_cache():
    return load_model(MODEL_PATH, compile=False)

model = load_model_cache()

# ---------- APP UI ----------
st.title("🥩 FreshLens AI: Quality Assessment for Meat Products")
st.write("Upload an image of meat and the AI will predict its freshness.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # ---------- DISPLAY IMAGE ----------
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # ---------- PREPROCESS ----------
    img_resized = img.resize(IMAGE_SIZE)
    img_array = img_to_array(img_resized)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ---------- PREDICTION ----------
    pred_probs = model.predict(img_array, verbose=0)[0]

    max_conf = np.max(pred_probs)

    if max_conf < 0.70:
        pred_class = "Uncertain"
        pred_class_index = np.argmax(pred_probs)
    else:
        pred_class_index = np.argmax(pred_probs)
        pred_class = CLASS_NAMES[pred_class_index]

    # ---------- DISPLAY RESULT ----------
    st.subheader(f"Prediction: **{pred_class}**")

    if pred_class != "Uncertain":
        st.write(f"Confidence: **{pred_probs[pred_class_index] * 100:.2f}%**")
    else:
        st.write(
            f"Highest Confidence: **{pred_probs[pred_class_index] * 100:.2f}%** "
            "(Below confidence threshold)"
        )

    # ---------- BAR CHART ----------
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(CLASS_NAMES, pred_probs, color=CLASS_COLORS, height=0.5)

    for i, v in enumerate(pred_probs):
        ax.text(
            v + 0.02,
            i,
            f"{v * 100:.2f}%",
            va='center',
            fontweight='bold'
        )

    ax.set_xlim(0, 1.1)
    ax.set_xticks([])
    ax.set_title("Prediction Confidence")
    st.pyplot(fig)

    # ---------- REASONING ----------
    reasoning = {
        "Fresh": (
            "✅ Color: Bright and natural\n"
            "✅ Texture: Firm and moist\n"
            "✅ Odor: Minimal or none\n"
            "➡ Action: Safe to consume or store properly"
        ),

        "Half-Fresh": (
            "⚠ Color: Slight discoloration\n"
            "⚠ Texture: Slightly soft or sticky\n"
            "⚠ Odor: Mild smell\n"
            "➡ Action: Consume soon; cook immediately"
        ),

        "Spoiled": (
            "❌ Color: Green/gray/brown\n"
            "❌ Texture: Slimy or mushy\n"
            "❌ Odor: Strong unpleasant smell\n"
            "➡ Action: Discard immediately"
        ),

        "Uncertain": (
            "⚠ The model is not confident enough\n"
            "⚠ Prediction is ambiguous between freshness classes\n"
            "⚠ Please inspect the meat manually\n"
            "➡ Upload a clearer image for better results"
        )
    }

    reasoning_colors = {
        "Fresh": "#d4edda",
        "Half-Fresh": "#fff3cd",
        "Spoiled": "#f8d7da",
        "Uncertain": "#d1ecf1"
    }

    reason_text = reasoning[pred_class].replace("\n", "<br>")

    st.markdown(
        f"""
        <div style="
            background-color:{reasoning_colors[pred_class]};
            padding:15px;
            border-radius:10px;
            font-weight:bold;
            color:#000;">
            {reason_text}
        </div>
        """,
        unsafe_allow_html=True
    )
