# llm/openai_client.py
from openai import OpenAI

class OpenAIExtractor:
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Wrapper for OpenAI API.
        Args:
            model: Model name (e.g. "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo")
        """
        self.client = OpenAI()
        self.model = model

    def extract(self, text: str, prompt_template: str) -> str:
        """Send text + prompt to OpenAI model"""
        prompt = prompt_template.format(text=text)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[OpenAIExtractor] Error: {e}")
            return "ERROR"
