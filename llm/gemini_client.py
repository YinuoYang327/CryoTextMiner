# clients/gemini_client.py
import base64
import json
import google.generativeai as genai
from PIL import Image
import io

def init_gemini_client(api_key: str):
    """
    Initialize Gemini client using API key.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-pro")
    return model


def analyze_image_gemini(model, image_path, prompt):
    """
    Given a Gemini model instance, image path, and text prompt,
    returns the list of predicted structures.
    """
    # read images 
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image = Image.open(io.BytesIO(image_bytes))

    try:
        response = model.generate_content([prompt, image])
        text = response.text.strip()
    except Exception as e:
        text = f"ERROR: {e}"

    # JSON output 
    try:
        structures = json.loads(text)
    except Exception:
        # if not standard JSON, then return the original text
        structures = [text]

    return structures
