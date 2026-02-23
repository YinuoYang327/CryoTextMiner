import base64
import json
import anthropic

def init_claude_client(api_key: str):
    """
    Initialize the Anthropic Claude client.
    """
    return anthropic.Anthropic(api_key=api_key)

def analyze_sequence_claude(client, image_paths, prompt_text):
    """
    Sequence inference for Claude 3.5 Sonnet.
    Accepts multiple images to analyze structural continuity.
    """
    content_list = []
    
    try:
        # Construct multi-image content blocks for Claude
        for path in image_paths:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
                content_list.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64
                    }
                })
        
        # Append the analytical prompt at the end of the image sequence
        content_list.append({"type": "text", "text": prompt_text})

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            temperature=0,
            messages=[{"role": "user", "content": content_list}]
        )
        text = response.content[0].text.strip()
    except Exception as e:
        text = f"ERROR: {e}"

    try:
        clean_text = text.replace('```json', '').replace('```', '').strip()
        return json.loads(clean_text)
    except Exception:
        return [text]