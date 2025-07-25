"""
Environment and configuration management utilities
"""

import os
import streamlit as st
from pathlib import Path
from typing import Dict, Any


def load_environment():
    """Load environment variables from .env file"""
    try:
        from dotenv import load_dotenv
        # Look for .env in config directory (parent directory)
        env_path = Path(__file__).parent.parent.parent / "config" / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            print(f"✅ Loaded environment from {env_path}")
            return True
        else:
            # Fallback to current directory
            load_dotenv()
            print("✅ Loaded environment from current directory")
            return True
    except ImportError:
        print("⚠️ python-dotenv not found, environment variables may not load properly")
        return False


def setup_environment():
    """Set up environment variables, respecting .env configuration"""
    # Database configuration
    if not os.getenv('DATABASE_URL'):
        os.environ["DATABASE_URL"] = "postgresql://postgres:2336@localhost:5432/pagila"
    
    # AI Provider is already loaded from .env file via load_dotenv()
    # Just ensure some defaults are set if not present
    ai_provider = os.getenv('AI_PROVIDER', 'ollama').lower()
    
    if ai_provider == 'ollama':
        # Ensure Ollama defaults are set
        os.environ.setdefault("OLLAMA_HOST", "http://localhost:11434")
        os.environ.setdefault("OLLAMA_MODEL", "mistral:7b-instruct")
        os.environ.setdefault("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")
    elif ai_provider == 'openai':
        # Ensure OpenAI defaults are set
        os.environ.setdefault("OPENAI_MODEL", "gpt-4o-mini")
    elif ai_provider == 'gemini':
        # Ensure Gemini defaults are set
        os.environ.setdefault("GEMINI_MODEL", "gemini-pro")
    elif ai_provider == 'openrouter':
        # Ensure OpenRouter defaults are set
        os.environ.setdefault("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-0324:free")


def check_environment() -> Dict[str, Any]:
    """Check and return environment status"""
    env_status = {}
    
    # Check database URL
    db_url = os.getenv('DATABASE_URL')
    env_status['database'] = {
        'status': 'OK' if db_url else 'MISSING',
        'value': db_url[:50] + '...' if db_url and len(db_url) > 50 else db_url
    }
    
    # Check AI provider using new AI_PROVIDER system
    ai_provider = os.getenv('AI_PROVIDER', 'ollama').lower()
    
    if ai_provider == 'openai':
        # OpenAI settings
        openai_key = os.getenv('OPENAI_API_KEY')
        openai_model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        
        env_status['ai_provider'] = {
            'type': 'OpenAI',
            'api_key': 'SET' if openai_key else 'MISSING',
            'model': openai_model,
            'provider': ai_provider
        }
    
    elif ai_provider == 'gemini':
        # Gemini settings
        gemini_key = os.getenv('GEMINI_API_KEY')
        gemini_model = os.getenv('GEMINI_MODEL', 'gemini-pro')
        
        env_status['ai_provider'] = {
            'type': 'Google Gemini',
            'api_key': 'SET' if gemini_key else 'MISSING',
            'model': gemini_model,
            'provider': ai_provider
        }
    
    elif ai_provider == 'openrouter':
        # OpenRouter settings
        openrouter_key = os.getenv('OPENROUTER_API_KEY')
        openrouter_model = os.getenv('OPENROUTER_MODEL', 'deepseek/deepseek-chat-v3-0324:free')
        
        env_status['ai_provider'] = {
            'type': 'OpenRouter',
            'api_key': 'SET' if openrouter_key else 'MISSING',
            'model': openrouter_model,
            'provider': ai_provider
        }
    
    elif ai_provider == 'ollama':
        # Ollama settings
        ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        ollama_model = os.getenv('OLLAMA_MODEL', 'mistral:7b-instruct')
        ollama_embed = os.getenv('OLLAMA_EMBEDDING_MODEL', 'mxbai-embed-large')
        
        env_status['ai_provider'] = {
            'type': 'Ollama (Local)',
            'host': ollama_host,
            'chat_model': ollama_model,
            'embedding_model': ollama_embed,
            'provider': ai_provider
        }
    
    else:
        env_status['ai_provider'] = {
            'type': 'UNKNOWN',
            'status': f'Unknown AI provider: {ai_provider}',
            'provider': ai_provider
        }
    
    return env_status
