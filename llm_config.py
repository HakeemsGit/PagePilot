import os
from typing import Dict, Type
from langchain_community.chat_models import ChatOpenAI, ChatAnthropic
# from langchain_google_genai import ChatGoogleGenerativeAI

LLM_CONFIGS: Dict[str, Dict] = {
    "openai": {
        "name": "OpenAI GPT-3.5",
        "class": ChatOpenAI,
        "kwargs": {
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.7
        }
    },
    "claude": {
        "name": "Anthropic Claude",
        "class": ChatAnthropic,
        "kwargs": {
            "model": "claude-2",
            "temperature": 0.7
        }
    },
    # "gemini": {
    #     "name": "Google Gemini Pro",
    #     "class": ChatGoogleGenerativeAI,
    #     "kwargs": {
    #         "model": "gemini-pro",
    #         "temperature": 0.7
    #     }
    # },
    "deepseek": {
        "name": "DeepSeek Chat",
        "class": ChatOpenAI,
        "kwargs": {
            "model": "deepseek-chat",
            "temperature": 0.7,
            "base_url": "https://api.deepseek.com/v1",
            "api_key": os.getenv("DEEPSEEK_API_KEY")
        }
    }
}
