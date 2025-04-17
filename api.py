from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
import torch
from uuid import uuid4
import os
from peft import PeftModel

app = FastAPI()

# Create outputs folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Mount static files so /outputs/image.png works
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Load model (on CPU)
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float32
).to("cpu")

# Load LoRA fine-tuned UNet weights
pipe.unet = PeftModel.from_pretrained(pipe.unet, "fashion-lora-sd-turbo-unet-data2x")

# Switch to eval
pipe.unet.eval()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate/")
async def generate_image(prompt_request: PromptRequest):
    try:
        image = pipe(prompt_request.prompt, num_inference_steps=10).images[0]
        filename = f"outputs/image_{uuid4().hex}.png"
        image.save(filename)
        return {"image_path": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
