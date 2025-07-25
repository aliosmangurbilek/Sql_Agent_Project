#!/usr/bin/env python3
"""
Pagila AI Assistant Pro - Application Launcher
This script launches the Streamlit app with proper environment configuration.
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Load environment variables from config folder
try:
    from dotenv import load_dotenv
    config_path = project_root / "config" / ".env"
    if config_path.exists():
        load_dotenv(config_path)
        print(f"✅ Loaded environment from {config_path}")
    else:
        print(f"⚠️ Environment file not found at {config_path}")
except ImportError:
    print("⚠️ python-dotenv not found, environment variables may not load properly")

def check_ollama():
    """Quick check if Ollama is running."""
    ai_provider = os.getenv('AI_PROVIDER', 'ollama').lower()
    if ai_provider != 'ollama':
        return True
    
    print("🦙 Checking Ollama...")
    try:
        import requests
        ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        response = requests.get(f"{ollama_host}/api/tags", timeout=3)
        if response.status_code == 200:
            print("✅ Ollama is running")
            return True
    except:
        pass
    
    print("⚠️ Ollama not accessible. Install from: https://ollama.ai/download")
    print("💡 Then run: ollama serve")
    return False

if __name__ == "__main__":
    import subprocess
    
    print("🎬 Pagila AI Assistant Pro - Quick Launcher")
    print("=" * 50)
    
    # Quick Ollama check
    check_ollama()
    
    # Change to src directory for proper imports
    os.chdir(src_path)
    
    # Launch Streamlit app
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        "app_pro.py", "--server.port", "8502"
    ]
    
    print("🚀 Starting Pagila AI Assistant Pro...")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🌐 URL: http://localhost:8502")
    print(f"🦙 AI Provider: {os.getenv('AI_PROVIDER', 'ollama')} (Free local AI)")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("💡 Try: python scripts/run_app.py")