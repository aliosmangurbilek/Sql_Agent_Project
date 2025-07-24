#!/usr/bin/env python3
"""
Quick launcher script for Pagila AI Assistant Pro
This script will check prerequisites and launch the Streamlit app.
"""

import os
import sys
import subprocess
import time

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
        print("  💡 Make sure to start the database with: cd pagila && docker-compose up -d")
        return False

def launch_streamlit():
    """Launch the Streamlit application."""
    print("🚀 Launching Streamlit application...")
    print("📱 Your app will be available at: http://localhost:8502")
    print("🔄 Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        # Launch streamlit with proper command
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
    
    # Check current directory
    if not os.path.exists("app_pro.py"):
        print("❌ app_pro.py not found in current directory")
        print("💡 Please run this script from the project root directory")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Missing dependencies. Please install required packages:")
        print("pip install streamlit flask flask-cors plotly pandas psycopg2-binary aiohttp")
        sys.exit(1)
    
    # Check database (optional, app can run without it for testing)
    db_ok = check_database()
    if not db_ok:
        print("\n⚠️  Database not available, but app will still launch")
        print("🔄 Some features may not work without database connection")
        time.sleep(2)
    
    print("\n✅ All checks passed!")
    print("🚀 Starting application...")
    time.sleep(1)
    
    # Launch the app
    launch_streamlit()

if __name__ == "__main__":
    main()
