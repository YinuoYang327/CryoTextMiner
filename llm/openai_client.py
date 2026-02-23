import base64
import json
from openai import OpenAI

def init_openai_client(api_key: str):
    """
    Initialize the OpenAI client with the provided API key.
    """
    return OpenAI(api_key=api_key)

def analyze_image_openai(client, image_path, prompt_text):
    """
    Inference function for GPT-4o.
    Uses base64 encoding for image transmission.
    """
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    b64 = base64.b64encode(img_bytes).decode("utf-8")

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert in cryo-electron tomography and cell biology."
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                    ]
                }
            ],
            temperature=0, # Crucial for coordinate precision
            max_tokens=1000
        )
        text = response.choices[0].message.content.strip()
    except Exception as e:
        text = f"ERROR: {e}"

    try:
        clean_text = text.replace('```json', '').replace('```', '').strip()
        structures = json.loads(clean_text)
    except Exception:
        structures = [text]
    return structures