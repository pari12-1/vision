# llm.py
import os
import random
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

STYLES = [
    "luxury fashion brand",
    "budget-friendly online store",
    "modern minimalist brand",
    "premium lifestyle brand",
    "trendy youth fashion label",
    "traditional ethnic brand"
]

def generate_product_details(category: str, confidence: float) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    style = random.choice(STYLES)

    prompt = f"""
You are a professional e-commerce copywriter for a {style}.

Product category: {category}
AI confidence score: {confidence:.2f}

Write a UNIQUE, non-repetitive, engaging product description of about 150 words.

Rules:
- Do NOT repeat phrasing from previous descriptions
- Vary sentence structure and tone
- Mention material, comfort, durability, styling tips
- Explain who this product is ideal for
- Make it sound natural and human (not AI-generated)
- Do NOT mention AI, model, or confidence

Make the description feel fresh and different.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You write creative, human-like product descriptions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.9,      # 🔥 more creativity
        top_p=0.95
    )

    return response.choices[0].message.content.strip()
