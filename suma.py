import streamlit as st
from tensorflow.keras.models import load_model

MODEL_PATH = r"model_v1_meat.h5"

try:
    model = load_model(MODEL_PATH)
except OSError as e:
    st.error(f"Failed to load model: {e}")
    st.stop()  # stop execution if model cannot be loaded
