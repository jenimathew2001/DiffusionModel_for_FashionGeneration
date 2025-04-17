# file: app.py

import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import torch
from diffusers import StableDiffusionPipeline

st.set_page_config(page_title="SD-Turbo Chatbot", layout="centered")

st.title("🧵 FashionBot powered by SD-Turbo")
st.caption("Enter a concept and get a unique fashion outfit with a designer-style sketch!")

# -------------------------------
# 🔍 Load model (for info only)
# -------------------------------
@st.cache_resource
def load_model():
    model = StableDiffusionPipeline.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float32)
    model.to("cpu")  # Load on CPU just for stats
    return model

pipe = load_model()

def get_model_info(model):
    total_params = sum(p.numel() for p in model.unet.parameters())
    return {
        "Total Parameters (UNet)": f"{total_params / 1e6:.2f}M",
        "Device": next(model.unet.parameters()).device.type.upper(),
        "Model": "stabilityai/sd-turbo"
    }

model_info = get_model_info(pipe)

# 🧠 Display model info
with st.expander("🧠 Model Info"):
    for k, v in model_info.items():
        st.write(f"**{k}:** {v}")

# -------------------------------
# 🎨 Prompt Input
# -------------------------------
prompt_input = st.text_input("🎨 Enter your fashion idea:")

if prompt_input:
    with st.spinner("🧠 Thinking like a fashion designer..."):
        try:
            # Send prompt to FastAPI backend
            response = requests.post(
                "http://127.0.0.1:8000/generate/",
                json={"prompt": prompt_input}
            )

            if response.status_code == 200:
                image_path = response.json()["image_path"]
                image_url = f"http://127.0.0.1:8000/{image_path}"

                # Fetch and show image
                image_response = requests.get(image_url)
                if image_response.status_code == 200:
                    img = Image.open(BytesIO(image_response.content))
                    st.image(img, caption="🧥 Your fashion concept")
                else:
                    st.error("Failed to load image from server.")
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
