import base64
import json
from openai import OpenAI

def init_openai_client(api_key: str):
    """Initialize OpenAI client with the given API key."""
    return OpenAI(api_key=api_key)

def analyze_image_openai(client, image_path, prompt):
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    b64 = base64.b64encode(img_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in cell biology and electron microscopy."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]}
        ],
        max_tokens=300
    )

    text = response.choices[0].message.content.strip()
    try:
        structures = json.loads(text)
    except Exception:
        structures = [text]
    return structures
