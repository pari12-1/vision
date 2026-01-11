# predict.py

import json
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

from .llm import generate_product_details

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CLASSES_PATH = BASE_DIR / "classes.json"

# -------------------------------------------------
# Load class names (loaded once at import time)
# -------------------------------------------------
with open(CLASSES_PATH, "r") as f:
    classes = json.load(f)

# -------------------------------------------------
# Image transform
# -------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# -------------------------------------------------
# Price mapping
# -------------------------------------------------
PRICE_MAP = {
    "shoes": 250,
    "watches": 100,
    "clothing": 450,
    "accessories": 300,
    "beauty": 200,
    "electronic": 1000,
    "toys": 150,
}

# -------------------------------------------------
# Default (non-LLM) text
# -------------------------------------------------
def generate_default_text(category: str):
    title = f"Premium {category.title()}"
    description = (
        f"This is a high-quality {category} designed for daily use, "
        f"offering durability, comfort, and reliable performance."
    )
    price = PRICE_MAP.get(category, 50)
    return title, description, price


# -------------------------------------------------
# Prediction function
# -------------------------------------------------
def predict_image(model, image_path: str):
    """
    Run image classification and generate product details.
    """

    if not Path(image_path).exists():
        raise FileNotFoundError("Image file not found")

    model.eval()

    # Ensure tensor & model are on same device
    device = next(model.parameters()).device

    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        confidence, idx = torch.max(probs, dim=1)

    confidence = confidence.item()
    category = classes[idx.item()]

    # -------------------------------------------------
    # Low-confidence fallback
    # -------------------------------------------------
    if confidence < 0.6:
        return {
            "category": "unknown",
            "confidence": round(confidence * 100, 2),
            "title": "Unknown Product",
            "description": "The product could not be confidently identified.",
            "price": 0,
            "llm_details": "AI could not generate details due to low confidence.",
        }

    # -------------------------------------------------
    # Default info
    # -------------------------------------------------
    title, default_description, price = generate_default_text(category)

    # -------------------------------------------------
    # LLM-generated description
    # -------------------------------------------------
    try:
        llm_description = generate_product_details(
            category=category,
            confidence=confidence,
        )
    except Exception:
        llm_description = "AI-generated product details are temporarily unavailable."

    # -------------------------------------------------
    # Final response
    # -------------------------------------------------
    return {
        "category": category,
        "title": title,
        "description": default_description,
        "price": price,
        "confidence": round(confidence * 100, 2),
        "llm_details": llm_description,
    }
