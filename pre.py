import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# ---------- CONFIG ----------
st.set_page_config(page_title="FreshLens AI: Quality Assessment for Meat Products", layout="centered")
MODEL_PATH = "model_v1_meat.h5"  # replace with your model path
IMAGE_SIZE = (128, 128)
CLASS_NAMES = ['Fresh', 'Half-Fresh', 'Spoiled']
CLASS_COLORS = ['green', 'orange', 'red']

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model_cache():
    return load_model(MODEL_PATH)

model = load_model_cache()

# ---------- APP ----------
st.title("🥩 FreshLens AI: Quality Assessment for Meat Products")
st.write("Upload an image of meat and the AI will predict its freshness.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img_array = img.resize(IMAGE_SIZE)
    img_array = img_to_array(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    pred_probs = model.predict(img_array)[0]
    pred_probs = np.round(pred_probs, 2)
    pred_class = CLASS_NAMES[np.argmax(pred_probs)]

    # Display prediction
    st.subheader(f"Prediction: **{pred_class}**")

    # Colored Bar chart
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(CLASS_NAMES, [1, 1, 1], color='white', edgecolor='gray', linewidth=2, height=0.5)
    ax.barh(CLASS_NAMES, pred_probs, color=CLASS_COLORS, height=0.5)

    # Add percentages on top of bars
    for index, value in enumerate(pred_probs):
        ax.text(value + 0.02, index, f"{100*value:.2f}%", fontsize=12, fontweight='bold', va='center')

    ax.set_xlim(0, 1.1)
    ax.set_xticks([])
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_yticklabels(CLASS_NAMES, fontweight='bold')
    st.pyplot(fig)

    # ---------------- ENHANCED REASONING ----------------
    reasoning = {
        "Fresh": (
            "✅ Color: Bright and natural.\n"
            "✅ Texture: Firm and moist.\n"
            "✅ Odor: Minimal or no odor.\n"
            "➡ Action: Safe to consume immediately or store properly."
        ),
        "Half-Fresh": (
            "⚠ Color: Slight discoloration.\n"
            "⚠ Texture: Softer or slightly sticky.\n"
            "⚠ Odor: Mild smell may be present.\n"
            "➡ Action: Consume soon; cook immediately. Avoid long storage."
        ),
        "Spoiled": (
            "❌ Color: Green/gray/brown discoloration.\n"
            "❌ Texture: Slimy or mushy.\n"
            "❌ Odor: Strong, unpleasant smell.\n"
            "➡ Action: Do not consume. Discard immediately to avoid food poisoning."
        )
    }

    reasoning_colors = {
        "Fresh": "#d4edda",      # light green
        "Half-Fresh": "#fff3cd", # light yellow
        "Spoiled": "#f8d7da"     # light red/pink
    }

    # Convert \n to <br> for HTML formatting
    reason_text = reasoning[pred_class].replace('\n', '<br>')

    # Display reasoning with colored background
    st.markdown(
        f"""
        <div style="background-color: {reasoning_colors[pred_class]};
                    padding: 15px;
                    border-radius: 10px;
                    font-weight: bold;
                    color: #000000;">
            {reason_text}
        </div>
        """,
        unsafe_allow_html=True
    )
