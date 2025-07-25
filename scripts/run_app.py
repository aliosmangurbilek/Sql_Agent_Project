#!/usr/bin/env python3
"""
Quick launcher script for Pagila AI Assistant Pro
This script will check prerequisites and launch the Streamlit app.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Get project root directory (parent of scripts folder)
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
config_path = project_root / "config"

def check_dependencies():
    """Check if required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    try:
        import streamlit
        print("  ✅ Streamlit installed")
    except ImportError:
        print("  ❌ Streamlit not found. Install with: pip install streamlit")
        return False
    
    try:
        import pandas
        print("  ✅ Pandas installed")
    except ImportError:
        print("  ❌ Pandas not found. Install with: pip install pandas")
        return False
    
    try:
        import plotly
        print("  ✅ Plotly installed")
    except ImportError:
        print("  ❌ Plotly not found. Install with: pip install plotly")
        return False
    
    try:
        import psycopg2
        print("  ✅ psycopg2 installed")
    except ImportError:
        print("  ❌ psycopg2 not found. Install with: pip install psycopg2-binary")
        return False
    
    return True

def check_database():
    """Check if database is accessible."""
    print("🗄️ Checking database connection...")
    
    try:
        import psycopg2
        conn = psycopg2.connect("postgresql://postgres:2336@localhost:5432/pagila")
        conn.close()
        print("  ✅ Database connection successful")
        return True
    except Exception as e:
        print(f"  ❌ Database connection failed: {str(e)[:50]}...")
        print("  💡 Make sure to start the database with: cd data/pagila && docker-compose up -d")
        return False

def check_ollama():
    """Check if Ollama is running (for local AI provider)."""
    from dotenv import load_dotenv
    load_dotenv(config_path / ".env")
    
    ai_provider = os.getenv('AI_PROVIDER', 'ollama').lower()
    if ai_provider != 'ollama':
        return True  # Not using Ollama, so no need to check
    
    print("🦙 Checking Ollama service...")
    
    try:
        import aiohttp
        import asyncio
        
        async def test_ollama():
            ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{ollama_host}/api/tags", timeout=5) as response:
                        if response.status == 200:
                            return True
                return False
            except:
                return False
        
        # Run the async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        is_running = loop.run_until_complete(test_ollama())
        loop.close()
        
        if is_running:
            print("  ✅ Ollama service is running")
            # Check if the required model is available
            ollama_model = os.getenv('OLLAMA_MODEL', 'mistral:7b-instruct')
            print(f"  📋 Using model: {ollama_model}")
            print("  💡 If model not found, run: ollama pull mistral:7b-instruct")
            return True
        else:
            print("  ❌ Ollama service not accessible")
            print("  💡 Start Ollama: ollama serve")
            print("  💡 Install Ollama: https://ollama.ai/download")
            return False
            
    except ImportError:
        print("  ⚠️  aiohttp not found, skipping Ollama check")
        print("  💡 Install with: pip install aiohttp")
        return True
    except Exception as e:
        print(f"  ⚠️  Ollama check failed: {str(e)[:50]}...")
        print("  💡 Make sure Ollama is installed and running: ollama serve")
        return False

def launch_streamlit():
    """Launch the Streamlit application."""
    print("🚀 Launching Streamlit application...")
    print("📱 Your app will be available at: http://localhost:8502")
    print("🔄 Press Ctrl+C to stop the application")
    print("-" * 50)
    
    # Load environment variables
    env_file = config_path / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
            print(f"✅ Loaded environment from {env_file}")
        except ImportError:
            print("⚠️ python-dotenv not found, environment variables may not load")
    
    try:
        # Change to src directory and launch streamlit
        os.chdir(src_path)
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "app_pro.py", "--server.port", "8502", "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Failed to launch Streamlit: {e}")

def main():
    """Main launcher function."""
    print("🎬 Pagila AI Assistant Pro - Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory structure
    if not src_path.exists():
        print(f"❌ src directory not found at {src_path}")
        print("💡 Please run this script from the project root or scripts directory")
        sys.exit(1)
    
    if not (src_path / "app_pro.py").exists():
        print(f"❌ app_pro.py not found in {src_path}")
        print("💡 Please ensure the project structure is correct")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        print(f"\n❌ Missing dependencies. Please install required packages:")
        print(f"pip install -r {config_path}/requirements.txt")
        sys.exit(1)
    
    # Check database (optional, app can run without it for testing)
    db_ok = check_database()
    if not db_ok:
        print("\n⚠️  Database not available, but app will still launch")
        print("🔄 Some features may not work without database connection")
        time.sleep(1)
    
    # Check Ollama if it's the selected provider
    ollama_ok = check_ollama()
    if not ollama_ok:
        print("\n⚠️  Ollama not available")
        print("🔄 You can still run the app, but AI features may not work")
        print("💡 Install Ollama from: https://ollama.ai/download")
        time.sleep(2)
    
    print("\n✅ All checks passed!")
    print("🚀 Starting application...")
    time.sleep(1)
    
    # Launch the app
    launch_streamlit()

if __name__ == "__main__":
    main()
