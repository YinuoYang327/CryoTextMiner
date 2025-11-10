import os
from dotenv import load_dotenv

# load .env
load_dotenv()

# get API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


from llm.openai_client import init_openai_client
from llm.gemini_client import init_gemini_client
from llm.claude_client import init_claude_client

openai_client = init_openai_client(OPENAI_API_KEY)
gemini_model = init_gemini_client(GEMINI_API_KEY)
claude_client = init_claude_client(ANTHROPIC_API_KEY)

from run import run_all_models
from evaluate_vlm_results import evaluate_results

if __name__ == "__main__":
    run_all_models(
        openai_client=openai_client,
        gemini_model=gemini_model,
        claude_client=claude_client,
        prompt_path="prompts/simple_prompt.txt"
    )
    evaluate_results("results/all_models_summary.csv")

