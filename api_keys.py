import os
from typing import Dict
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv(override=True)

def get_api_key(provider: str) -> str:
    """Get API key for a provider from environment variables"""
    env_mapping = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY"
    }
    
    if provider not in env_mapping:
        raise ValueError(f"Unknown provider: {provider}")
        
    return os.getenv(env_mapping[provider], "")

def set_api_key(provider: str, key: str) -> None:
    """Set API key for a provider in environment"""
    env_mapping = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY"
    }
    
    if provider not in env_mapping:
        raise ValueError(f"Unknown provider: {provider}")
        
    os.environ[env_mapping[provider]] = key
