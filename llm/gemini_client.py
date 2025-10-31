# llm/gemini_client.py
import google.generativeai as genai
import os

class GeminiExtractor:
    def __init__(self, model: str = "gemini-1.5-pro"):
        """
        Wrapper for Google Gemini API.
        Requires: export GOOGLE_API_KEY="your_key"
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def extract(self, text: str, prompt_template: str) -> str:
        """Send text + prompt to Gemini model"""
        prompt = prompt_template.format(text=text)
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"[GeminiExtractor] Error: {e}")
            return "ERROR"
