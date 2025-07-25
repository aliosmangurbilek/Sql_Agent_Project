#!/usr/bin/env python3
"""
Simple AI Provider Test for Streamlit
"""

import streamlit as st
import os
import re
from dotenv import load_dotenv
from pathlib import Path

# Load environment from config directory
env_path = Path(__file__).parent.parent / "config" / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

st.title("🤖 AI Provider Manager")

st.markdown("### Current Configuration")
current_provider = os.getenv('AI_PROVIDER', 'ollama')
st.info(f"Active Provider: **{current_provider.upper()}**")

# Display current provider status
if current_provider == 'openai':
    key_status = "✅ SET" if os.getenv('OPENAI_API_KEY') else "❌ MISSING"
    st.write(f"OpenAI API Key: {key_status}")
    st.write(f"Model: {os.getenv('OPENAI_MODEL', 'not set')}")
elif current_provider == 'gemini':
    key_status = "✅ SET" if os.getenv('GEMINI_API_KEY') else "❌ MISSING"
    st.write(f"Gemini API Key: {key_status}")
    st.write(f"Model: {os.getenv('GEMINI_MODEL', 'not set')}")
elif current_provider == 'openrouter':
    key_status = "✅ SET" if os.getenv('OPENROUTER_API_KEY') else "❌ MISSING"
    st.write(f"OpenRouter API Key: {key_status}")
    st.write(f"Model: {os.getenv('OPENROUTER_MODEL', 'not set')}")
elif current_provider == 'ollama':
    st.write(f"Ollama Host: {os.getenv('OLLAMA_HOST', 'not set')}")
    st.write(f"Model: {os.getenv('OLLAMA_MODEL', 'not set')}")

st.markdown("---")

# Provider selection
st.markdown("### Switch AI Provider")

provider_choice = st.selectbox(
    "Choose Provider:",
    ["ollama", "openai", "gemini", "openrouter"],
    index=["ollama", "openai", "gemini", "openrouter"].index(current_provider)
)

# Provider-specific settings
if provider_choice == "openai":
    st.markdown("#### OpenAI Configuration")
    api_key = st.text_input("API Key:", type="password", value=os.getenv('OPENAI_API_KEY', ''))
    model = st.selectbox("Model:", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o-mini"])
    
    if st.button("Apply OpenAI Settings"):
        # Update .env
        env_path = Path(__file__).parent.parent / "config" / ".env"
        with open(env_path, 'r') as f:
            content = f.read()
        
        content = re.sub(r'AI_PROVIDER=.*', f'AI_PROVIDER={provider_choice}', content)
        if api_key:
            content = re.sub(r'OPENAI_API_KEY=.*', f'OPENAI_API_KEY={api_key}', content)
        content = re.sub(r'OPENAI_MODEL=.*', f'OPENAI_MODEL={model}', content)
        
        with open(env_path, 'w') as f:
            f.write(content)
        
        st.success("✅ OpenAI settings applied!")

elif provider_choice == "gemini":
    st.markdown("#### Gemini Configuration")
    api_key = st.text_input("API Key:", type="password", value=os.getenv('GEMINI_API_KEY', ''))
    model = st.selectbox("Model:", ["gemini-pro", "gemini-pro-vision"])
    
    if st.button("Apply Gemini Settings"):
        env_path = Path(__file__).parent.parent / "config" / ".env"
        with open(env_path, 'r') as f:
            content = f.read()
        
        content = re.sub(r'AI_PROVIDER=.*', f'AI_PROVIDER={provider_choice}', content)
        if api_key:
            content = re.sub(r'GEMINI_API_KEY=.*', f'GEMINI_API_KEY={api_key}', content)
        content = re.sub(r'GEMINI_MODEL=.*', f'GEMINI_MODEL={model}', content)
        
        with open(env_path, 'w') as f:
            f.write(content)
        
        st.success("✅ Gemini settings applied!")

elif provider_choice == "openrouter":
    st.markdown("#### OpenRouter Configuration")
    api_key = st.text_input("API Key:", type="password", value=os.getenv('OPENROUTER_API_KEY', ''))
    
    st.markdown("**Free Models:**")
    free_models = [
        "deepseek/deepseek-chat-v3-0324:free",
        "meta-llama/llama-3.1-8b-instruct:free",
        "microsoft/phi-3-mini-128k-instruct:free"
    ]
    
    model = st.selectbox("Model:", free_models + [
        "openai/gpt-4-turbo",
        "anthropic/claude-3.5-sonnet",
        "google/gemini-pro-1.5"
    ])
    
    if st.button("Apply OpenRouter Settings"):
        env_path = Path(__file__).parent.parent / "config" / ".env"
        with open(env_path, 'r') as f:
            content = f.read()
        
        content = re.sub(r'AI_PROVIDER=.*', f'AI_PROVIDER={provider_choice}', content)
        if api_key:
            content = re.sub(r'OPENROUTER_API_KEY=.*', f'OPENROUTER_API_KEY={api_key}', content)
        content = re.sub(r'OPENROUTER_MODEL=.*', f'OPENROUTER_MODEL={model}', content)
        
        with open(env_path, 'w') as f:
            f.write(content)
        
        st.success("✅ OpenRouter settings applied!")

elif provider_choice == "ollama":
    st.markdown("#### Ollama Configuration")
    host = st.text_input("Host:", value=os.getenv('OLLAMA_HOST', 'http://localhost:11434'))
    model = st.selectbox("Model:", ["mistral:7b-instruct", "llama2", "codellama", "phi"])
    
    if st.button("Apply Ollama Settings"):
        env_path = Path(__file__).parent.parent / "config" / ".env"
        with open(env_path, 'r') as f:
            content = f.read()
        
        content = re.sub(r'AI_PROVIDER=.*', f'AI_PROVIDER={provider_choice}', content)
        content = re.sub(r'OLLAMA_HOST=.*', f'OLLAMA_HOST={host}', content)
        content = re.sub(r'OLLAMA_MODEL=.*', f'OLLAMA_MODEL={model}', content)
        
        with open(env_path, 'w') as f:
            f.write(content)
        
        st.success("✅ Ollama settings applied!")

st.markdown("---")
st.markdown("**Note:** After changing settings, refresh the main app to apply changes.")
