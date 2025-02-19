# File: benchmark.py
import os
from dotenv import load_dotenv
from main import run_benchmark
from test_data import TEST_DATA

# Load environment variables
load_dotenv()

# ----- USER CONFIGURABLE SECTION -----
TARGET_LANGUAGE = "Tagalog"

# API keys for different providers
API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "google": os.getenv("GEMINI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "deepseek": os.getenv("DEEPOSEEK_API_KEY"),
    "perplexity": os.getenv("PERPLEXITY_API_KEY"),
}

# LLM configurations: feel free to add or modify models here
LLM_CONFIGS = [
    {"provider": "openai", "model": "gpt-4o", "name": "GPT-4o"},
    {"provider": "openai", "model": "gpt-4o-mini", "name": "GPT-4o-mini"},
    {"provider": "openai", "model": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo"},
    {"provider": "openai", "model": "o1", "name": "o1"},
    {"provider": "google", "model": "gemini-2.0-flash", "name": "Gemini 2.0 Flash"},
    {"provider": "google", "model": "gemini-2.0-flash-lite-preview-02-05", "name": "Gemini 2.0 Flash Lite Preview"},
    {"provider": "anthropic", "model": "claude-3-5-haiku-20241022", "name": "Claude 3.5 Haiku"},
]
# ---------------------------------------

if __name__ == "__main__":
    run_benchmark(
        target_language=TARGET_LANGUAGE,
        llm_configs=LLM_CONFIGS,
        api_keys=API_KEYS,
        test_data=TEST_DATA,
        embedding_api_key=API_KEYS["openai"]  # Using OpenAI key for embeddings
    )