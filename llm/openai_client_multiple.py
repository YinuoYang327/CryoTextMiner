import base64
import json
from openai import OpenAI

def init_openai_client(api_key: str):
    """
    Initialize the OpenAI client with the provided API key.
    """
    return OpenAI(api_key=api_key)

def analyze_sequence_openai(client, image_paths, prompt_text):
    """
    Sequence inference for GPT-4o.
    Processes a list of image paths as a single combined visual context.
    """
    # Initialize message content with the text prompt
    content_list = [{"type": "text", "text": prompt_text}]

    try:
        # Encode each image in the sequence and add to the request payload
        for path in image_paths:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
                content_list.append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/png;base64,{b64}"}
                })

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in cryo-electron tomography."},
                {"role": "user", "content": content_list}
            ],
            temperature=0,
            max_tokens=1500
        )
        text = response.choices[0].message.content.strip()
    except Exception as e:
        text = f"ERROR: {e}"

    try:
        clean_text = text.replace('```json', '').replace('```', '').strip()
        return json.loads(clean_text)
    except Exception:
        return [text]