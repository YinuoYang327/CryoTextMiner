# clients/claude_client.py
import base64
import json
import anthropic

def init_claude_client(api_key: str):
    """
    Initialize Anthropic Claude client using API key.
    """
    return anthropic.Anthropic(api_key=api_key)


def analyze_image_claude(client, image_path, prompt):
    """
    Given a Claude client, image path, and prompt,
    returns a list of predicted structures.
    """
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    b64_image = base64.b64encode(img_bytes).decode("utf-8")

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": (
                            "You are a biomedical image analysis expert. "
                            "Identify the subcellular structures visible in this electron microscopy image.\n\n"
                            + prompt
                        )},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64_image
                            }
                        }
                    ]
                }
            ]
        )
        text = response.content[0].text.strip()
    except Exception as e:
        text = f"ERROR: {e}"

    try:
        structures = json.loads(text)
    except Exception:
        structures = [text]

    return structures
