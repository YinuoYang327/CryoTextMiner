# utils/prompt_manager.py
import re

def get_prompt_by_id(file_path, prompt_id):
    """
    Extracts a specific prompt from a collection file based on its ID.
    The collection file uses [PROMPT_ID] as markers.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Regex explanation:
        # \[{prompt_id}\] : Matches the opening tag like [SIMPLE_ID]
        # \n(.*?)          : Matches and captures everything after the tag
        # (?=\n\[|$)      : Lookahead to stop before the next tag or end of file
        pattern = rf"\[{prompt_id}\]\n(.*?)(?=\n\[|$)"
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            # Strip leading/trailing whitespace for a clean prompt
            return match.group(1).strip()
        else:
            available_ids = re.findall(r"\[(.*?)\]", content)
            raise ValueError(
                f"Prompt ID '{prompt_id}' not found. "
                f"Available IDs in file: {available_ids}"
            )
            
    except FileNotFoundError:
        print(f"ERROR: Prompt collection file not found at {file_path}")
        return None
    except Exception as e:
        print(f"ERROR: Failed to retrieve prompt: {e}")
        return None