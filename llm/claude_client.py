# llm/claude_client.py
import os
from anthropic import Anthropic

class ClaudeExtractor:
    def __init__(self, model: str = "claude-3-5-sonnet-20240620"):
        """
        Wrapper for Anthropic Claude API.
        Requires: export ANTHROPIC_API_KEY="your_key"
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set.")
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def extract(self, text: str, prompt_template: str) -> str:
        """Send text + prompt to Claude model"""
        prompt = prompt_template.format(text=text)
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"[ClaudeExtractor] Error: {e}")
            return "ERROR"
