import json
import google.generativeai as genai
from PIL import Image
import io

def init_gemini_client(api_key: str):
    """
    Initialize the Google Gemini client with the provided API key.
    """
    genai.configure(api_key=api_key)
    # Using 1.5-pro for best multimodal performance in scientific imaging
    model = genai.GenerativeModel("gemini-2.5-pro") 
    return model

def analyze_image_gemini(model, image_path, prompt_text):
    """
    Inference function for Gemini. 
    Accepts model instance, image path, and pre-loaded prompt string.
    """
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image = Image.open(io.BytesIO(image_bytes))

    try:
        # Temperature=0 ensures reproducible scientific results
        response = model.generate_content(
            [prompt_text, image],
            generation_config={"temperature": 0}
        )
        text = response.text.strip()
    except Exception as e:
        text = f"ERROR: {e}"

    # Clean Markdown JSON tags if present and parse
    try:
        clean_text = text.replace('```json', '').replace('```', '').strip()
        structures = json.loads(clean_text)
    except Exception:
        structures = [text]

    return structures