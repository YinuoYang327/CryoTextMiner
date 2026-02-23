import json
import google.generativeai as genai
from PIL import Image
import io

def init_gemini_client(api_key: str):
    """
    Initialize the Google Gemini client with the provided API key.
    """
    genai.configure(api_key=api_key)
    # Using 1.5-pro for best sequence reasoning in Cryo-ET slices
    model = genai.GenerativeModel("gemini-2.5-pro") 
    return model

def analyze_sequence_gemini(model, image_paths, prompt_text):
    """
    Inference function for a sequence of images.
    Accepts a list of image paths and a single prompt.
    """
    contents = [prompt_text]
    
    try:
        # Load all images in the sequence into the content list
        for path in image_paths:
            with open(path, "rb") as f:
                img = Image.open(io.BytesIO(f.read()))
                contents.append(img)

        # Generate content with temperature=0 for scientific consistency
        response = model.generate_content(
            contents,
            generation_config={"temperature": 0}
        )
        text = response.text.strip()
    except Exception as e:
        text = f"ERROR: {e}"

    # Standardize output to list/JSON format
    try:
        clean_text = text.replace('```json', '').replace('```', '').strip()
        return json.loads(clean_text)
    except Exception:
        return [text]