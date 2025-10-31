prompt_template = """
You are a biology assistant that identifies subcellular structures.

Input:
"{text}"

Output a JSON list of structures mentioned.
"""

text = "This EM image shows mitochondria and the endoplasmic reticulum."

from llm.openai_client import OpenAIExtractor
from llm.gemini_client import GeminiExtractor
from llm.claude_client import ClaudeExtractor

clients = [
    OpenAIExtractor("gpt-4o-mini"),
    GeminiExtractor("gemini-1.5-pro"),
    ClaudeExtractor("claude-3-5-sonnet-20240620")
]

for c in clients:
    print(f"\n=== {c.__class__.__name__} ===")
    print(c.extract(text, prompt_template))
