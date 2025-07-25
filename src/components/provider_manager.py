"""
AI Provider management component
"""

import streamlit as st
import os
import re
from pathlib import Path


def render_provider_selector():
    """Render AI provider selection interface"""
    st.markdown("### 🤖 AI Provider")
    
    # Current provider status
    current_provider = os.getenv('AI_PROVIDER', 'ollama').lower()
    
    # Provider status display
    if current_provider == 'openai':
        st.success("✅ Using OpenAI GPT for SQL generation")
        if not os.getenv('OPENAI_API_KEY'):
            st.error("❌ OPENAI_API_KEY not set!")
            st.code("Set OPENAI_API_KEY in .env file")
    elif current_provider == 'gemini':
        st.success("✅ Using Google Gemini for SQL generation")
        if not os.getenv('GEMINI_API_KEY'):
            st.error("❌ GEMINI_API_KEY not set!")
            st.code("Set GEMINI_API_KEY in .env file")
    elif current_provider == 'openrouter':
        st.success("✅ Using OpenRouter for SQL generation")
        if not os.getenv('OPENROUTER_API_KEY'):
            st.error("❌ OPENROUTER_API_KEY not set!")
            st.code("Set OPENROUTER_API_KEY in .env file")
    elif current_provider == 'ollama':
        st.info("🦙 Using Ollama for SQL generation")
    else:
        st.warning(f"⚠️ Unknown AI provider: {current_provider}")


def render_provider_switching():
    """Render provider switching interface"""
    with st.expander("🔧 Change AI Provider", expanded=False):
        provider_options = [
            "Ollama (Local, Free)",
            "OpenAI GPT (API Key Required)",
            "Google Gemini (API Key Required)",
            "OpenRouter (Multiple Models, API Key Required)"
        ]
        
        # Map current provider to index
        current_provider = os.getenv('AI_PROVIDER', 'ollama').lower()
        provider_index = 0  # default to Ollama
        if current_provider == 'openai':
            provider_index = 1
        elif current_provider == 'gemini':
            provider_index = 2
        elif current_provider == 'openrouter':
            provider_index = 3
        
        provider_choice = st.radio(
            "Select AI Provider:",
            provider_options,
            index=provider_index
        )
        
        # Provider-specific configuration
        if provider_choice == "OpenAI GPT (API Key Required)":
            _render_openai_config()
        elif provider_choice == "Google Gemini (API Key Required)":
            _render_gemini_config()
        elif provider_choice == "OpenRouter (Multiple Models, API Key Required)":
            _render_openrouter_config()
        else:  # Ollama
            _render_ollama_config()


def _render_openai_config():
    """Render OpenAI configuration"""
    st.info("💡 OpenAI GPT provides reliable SQL generation")
    api_key_input = st.text_input(
        "OpenAI API Key:", 
        type="password",
        placeholder="sk-...",
        value=os.getenv('OPENAI_API_KEY', '')
    )
    
    model_choice = st.selectbox(
        "Model:",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o-mini"],
        index=3 if os.getenv('OPENAI_MODEL') == 'gpt-4o-mini' else 0
    )
    
    if st.button("🔄 Switch to OpenAI"):
        if api_key_input:
            _update_env_file('openai', {
                'OPENAI_API_KEY': api_key_input,
                'OPENAI_MODEL': model_choice
            })
            st.success("✅ Switched to OpenAI! Please refresh the page.")
            st.rerun()
        else:
            st.error("Please enter your OpenAI API key")


def _render_gemini_config():
    """Render Gemini configuration"""
    st.info("🔮 Google Gemini provides advanced AI capabilities")
    api_key_input = st.text_input(
        "Gemini API Key:", 
        type="password",
        placeholder="AIza...",
        value=os.getenv('GEMINI_API_KEY', '')
    )
    
    model_choice = st.selectbox(
        "Model:",
        ["gemini-pro", "gemini-pro-vision", "gemini-1.5-pro"],
        index=0
    )
    
    if st.button("🔄 Switch to Gemini"):
        if api_key_input:
            _update_env_file('gemini', {
                'GEMINI_API_KEY': api_key_input,
                'GEMINI_MODEL': model_choice
            })
            st.success("✅ Switched to Gemini! Please refresh the page.")
            st.rerun()
        else:
            st.error("Please enter your Gemini API key")


def _render_openrouter_config():
    """Render OpenRouter configuration"""
    st.info("🌐 OpenRouter provides access to multiple AI models")
    api_key_input = st.text_input(
        "OpenRouter API Key:", 
        type="password",
        placeholder="sk-or-v1-...",
        value=os.getenv('OPENROUTER_API_KEY', '')
    )
    
    st.markdown("**Available Models:**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🆓 Free Models:**")
        free_models = [
            "meta-llama/llama-3.1-8b-instruct:free",
            "deepseek/deepseek-chat-v3-0324:free",
            "microsoft/phi-3-mini-128k-instruct:free"
        ]
        for model in free_models:
            st.code(model)
    
    with col2:
        st.markdown("**💎 Premium Models:**")
        premium_models = [
            "openai/gpt-4-turbo",
            "anthropic/claude-3.5-sonnet",
            "google/gemini-pro-1.5"
        ]
        for model in premium_models:
            st.code(model)
    
    current_model = os.getenv('OPENROUTER_MODEL', 'deepseek/deepseek-chat-v3-0324:free')
    model_choice = st.text_input(
        "Model:",
        value=current_model,
        placeholder="deepseek/deepseek-chat-v3-0324:free"
    )
    
    if st.button("🔄 Switch to OpenRouter"):
        if api_key_input:
            _update_env_file('openrouter', {
                'OPENROUTER_API_KEY': api_key_input,
                'OPENROUTER_MODEL': model_choice
            })
            st.success("✅ Switched to OpenRouter! Please refresh the page.")
            st.rerun()
        else:
            st.error("Please enter your OpenRouter API key")


def _render_ollama_config():
    """Render Ollama configuration"""
    st.info("🆓 Ollama is free but requires local installation")
    
    ollama_host = st.text_input(
        "Ollama Host:",
        value=os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    )
    
    model_choice = st.selectbox(
        "Model:",
        ["mistral:7b-instruct", "llama2", "codellama", "phi"],
        index=0
    )
    
    if st.button("🔄 Switch to Ollama"):
        _update_env_file('ollama', {
            'OLLAMA_HOST': ollama_host,
            'OLLAMA_MODEL': model_choice
        })
        st.success("✅ Switched to Ollama! Please refresh the page.")
        st.rerun()


def _update_env_file(provider: str, config: dict):
    """Update .env file with new provider configuration"""
    env_path = Path(__file__).parent.parent.parent / "config" / ".env"
    
    with open(env_path, 'r') as f:
        content = f.read()
    
    # Update AI_PROVIDER
    content = re.sub(r'AI_PROVIDER=.*', f'AI_PROVIDER={provider}', content)
    
    # Update provider-specific settings
    for key, value in config.items():
        if key in content:
            content = re.sub(f'{key}=.*', f'{key}={value}', content)
    
    with open(env_path, 'w') as f:
        f.write(content)
