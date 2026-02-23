# utils/config_loader.py

def load_api_keys(file_path="keys/api_keys.txt"):
    """
    Parses the API key file and returns a dictionary of keys.
    The file should follow the format: KEY_NAME=VALUE
    """
    keys = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                # Skip empty lines or lines without '='
                line = line.strip()
                if not line or "=" not in line:
                    continue
                
                # Split by the first '=' found
                name, value = line.split("=", 1)
                keys[name.strip()] = value.strip()
    except FileNotFoundError:
        print(f"ERROR: API key file not found at {file_path}")
    except Exception as e:
        print(f"ERROR: Failed to load API keys: {e}")
        
    return keys