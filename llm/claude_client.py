import base64
import json
import anthropic

def init_claude_client(api_key: str):
    """
    Initialize the Anthropic Claude client with the provided API key.
    """
    return anthropic.Anthropic(api_key=api_key)

def analyze_image_claude(client, image_path, prompt_text):
    """
    Inference function for Claude 3.5 Sonnet.
    Accepts system prompt instructions within the message body.
    """
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    b64_image = base64.b64encode(img_bytes).decode("utf-8")

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64_image
                            }
                        },
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]
        )
        text = response.content[0].text.strip()
    except Exception as e:
        text = f"ERROR: {e}"

    try:
        clean_text = text.replace('```json', '').replace('```', '').strip()
        structures = json.loads(clean_text)
    except Exception:
        structures = [text]

    return structures