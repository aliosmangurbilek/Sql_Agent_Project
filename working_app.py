#!/usr/bin/env python3
"""
Fully functional Streamlit app for Pagila AI Assistant Pro
Natural Language to SQL AI Agent
"""

import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time
from datetime import datetime
import json
import requests

# Try to import database and other dependencies
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    st.warning("⚠️ psycopg2 not installed. Install with: pip install psycopg2-binary")

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    st.warning("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Load environment variables
if DOTENV_AVAILABLE:
    project_root = Path(__file__).parent
    env_path = project_root / "config" / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        st.sidebar.success("✅ Environment loaded")
    else:
        st.sidebar.warning("⚠️ No .env file found")

def get_database_schema():
    """Get database schema for AI context"""
    schema_info = """
    PAGILA DATABASE SCHEMA:
    
    Tables:
    - film: title, description, release_year, language_id, rental_duration, rental_rate, length, replacement_cost, rating, special_features
    - actor: actor_id, first_name, last_name, last_update
    - category: category_id, name, last_update
    - customer: customer_id, store_id, first_name, last_name, email, address_id, activebool, create_date, last_update, active
    - rental: rental_id, rental_date, inventory_id, customer_id, return_date, staff_id, last_update
    - payment: payment_id, customer_id, staff_id, rental_id, amount, payment_date
    - inventory: inventory_id, film_id, store_id, last_update
    - store: store_id, manager_staff_id, address_id, last_update
    - staff: staff_id, first_name, last_name, address_id, email, store_id, active, username, password, last_update
    - address: address_id, address, address2, district, city_id, postal_code, phone, last_update
    - city: city_id, city, country_id, last_update
    - country: country_id, country, last_update
    - film_actor: actor_id, film_id, last_update
    - film_category: film_id, category_id, last_update
    - language: language_id, name, last_update
    
    Common relationships:
    - film ↔ film_actor ↔ actor
    - film ↔ film_category ↔ category
    - customer ↔ rental ↔ inventory ↔ film
    - customer ↔ payment
    - customer ↔ address ↔ city ↔ country
    """
    return schema_info

async def call_ollama_api(prompt: str, model: str = None) -> str:
    """Call Ollama API for natural language to SQL conversion"""
    try:
        ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        model = model or os.getenv('OLLAMA_MODEL', 'mistral:7b-instruct')
        
        url = f"{ollama_host}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 500
            }
        }
        
        # Increased timeout and better error handling
        response = requests.post(url, json=payload, timeout=240)
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', '').strip()
        else:
            return f"Error: Ollama API returned status {response.status_code}"
            
    except requests.exceptions.Timeout:
        return "Error: Request timed out. The model might be slow to respond. Try a simpler question or check if Ollama is running properly."
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Ollama. Make sure Ollama is running on the specified host."
    except Exception as e:
        return f"Error calling Ollama: {str(e)}"

def call_openai_api(prompt: str, model: str = None) -> str:
    """Call OpenAI API for natural language to SQL conversion"""
    try:
        if not OPENAI_AVAILABLE:
            return "Error: OpenAI library not installed"
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return "Error: OpenAI API key not found"
        
        client = OpenAI(api_key=api_key, timeout=120.0)  # Increased timeout
        model = model or os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a SQL expert. Convert natural language questions to PostgreSQL queries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        if "timeout" in str(e).lower():
            return "Error: OpenAI request timed out. Please try again with a simpler question."
        return f"Error calling OpenAI: {str(e)}"

def call_gemini_api(prompt: str, model: str = None) -> str:
    """Call Google Gemini API for natural language to SQL conversion"""
    try:
        if not GEMINI_AVAILABLE:
            return "Error: Google Generative AI library not installed"
        
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            return "Error: Gemini API key not found"
        
        genai.configure(api_key=api_key)
        model_name = model or os.getenv('GEMINI_MODEL', 'gemini-pro')
        model = genai.GenerativeModel(model_name)
        
        # Configure generation with timeout handling
        generation_config = genai.types.GenerationConfig(
            temperature=0.1,
            max_output_tokens=500,
        )
        
        response = model.generate_content(prompt, generation_config=generation_config)
        return response.text.strip()
        
    except Exception as e:
        if "timeout" in str(e).lower() or "deadline" in str(e).lower():
            return "Error: Gemini request timed out. Please try again with a simpler question."
        return f"Error calling Gemini: {str(e)}"

def call_openrouter_api(prompt: str, model: str = None) -> str:
    """Call OpenRouter API for natural language to SQL conversion"""
    try:
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            return "Error: OpenRouter API key not found"
        
        # Use provided model or fall back to environment default
        selected_model = model or os.getenv('OPENROUTER_MODEL', 'deepseek/deepseek-chat-v3-0324:free')
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/aliosmangurbilek/iga_staj_project",
            "X-Title": "Pagila AI Assistant Pro"
        }
        
        payload = {
            "model": selected_model,
            "messages": [
                {"role": "system", "content": "You are a SQL expert. Convert natural language questions to PostgreSQL queries."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 500
        }
        
        # Increased timeout and better error handling
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            return f"Error: OpenRouter API returned status {response.status_code}: {response.text}"
            
    except requests.exceptions.Timeout:
        return "Error: OpenRouter request timed out. The model might be busy. Please try again."
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to OpenRouter. Check your internet connection."
    except Exception as e:
        return f"Error calling OpenRouter: {str(e)}"

def natural_language_to_sql(question: str, provider: str = "ollama", model: str = None) -> dict:
    """Convert natural language question to SQL query using specified AI provider"""
    
    schema = get_database_schema()
    
    prompt = f"""
You are an expert SQL developer working with a PostgreSQL database called 'pagila' (a sample movie rental database).

DATABASE SCHEMA:
{schema}

USER QUESTION: "{question}"

Instructions:
1. Convert the user's natural language question into a valid PostgreSQL SQL query
2. Use proper table joins when needed
3. Include appropriate WHERE clauses, ORDER BY, and LIMIT statements
4. Return ONLY the SQL query, no explanations
5. Make sure the query is safe (no DROP, DELETE, UPDATE, INSERT statements)
6. If the question is unclear, create the most reasonable interpretation

Examples:
- "How many movies are there?" → SELECT COUNT(*) FROM film;
- "Top 5 longest movies" → SELECT title, length FROM film ORDER BY length DESC LIMIT 5;
- "Movies with Tom Cruise" → SELECT f.title FROM film f JOIN film_actor fa ON f.film_id = fa.film_id JOIN actor a ON fa.actor_id = a.actor_id WHERE a.first_name = 'TOM' AND a.last_name = 'CRUISE';

SQL Query:
"""

    try:
        if provider == "ollama":
            import asyncio
            response = asyncio.run(call_ollama_api(prompt, model))
        elif provider == "openai":
            response = call_openai_api(prompt, model)
        elif provider == "gemini":
            response = call_gemini_api(prompt, model)
        elif provider == "openrouter":
            response = call_openrouter_api(prompt, model)
        else:
            return {"error": f"Unknown provider: {provider}", "sql": None}
        
        # Extract SQL from response
        sql_query = extract_sql_from_response(response)
        
        return {
            "sql": sql_query,
            "raw_response": response,
            "provider": provider,
            "model": model,
            "error": None
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "sql": None,
            "raw_response": None,
            "provider": provider,
            "model": model
        }

def extract_sql_from_response(response: str) -> str:
    """Extract SQL query from AI response"""
    if not response:
        return ""
    
    # Common patterns to clean up the response
    lines = response.split('\n')
    sql_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and explanatory text
        if not line:
            continue
            
        # Skip lines that don't look like SQL
        if line.lower().startswith(('here', 'the query', 'this query', 'explanation', 'note:')):
            continue
            
        # Remove common prefixes
        if line.startswith('```sql'):
            continue
        if line.startswith('```'):
            if sql_lines:  # End of SQL block
                break
            continue
            
        # Look for SQL keywords
        if any(keyword in line.upper() for keyword in ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE']):
            sql_lines.append(line)
        elif sql_lines:  # Continue collecting SQL if we've started
            sql_lines.append(line)
    
    if sql_lines:
        sql = ' '.join(sql_lines)
        # Clean up the SQL
        sql = sql.strip()
        if not sql.endswith(';'):
            sql += ';'
        return sql
    
    # Fallback: try to extract anything that looks like SQL
    response = response.strip()
    if 'SELECT' in response.upper():
        # Find the SELECT statement
        start = response.upper().find('SELECT')
        sql_part = response[start:]
        
        # Clean up
        sql_part = sql_part.split('\n')[0] if '\n' in sql_part else sql_part
        sql_part = sql_part.replace('```', '').strip()
        
        if not sql_part.endswith(';'):
            sql_part += ';'
            
        return sql_part
    
    return response.strip()

def test_database_connection():
    """Test database connection"""
    if not DB_AVAILABLE:
        return False, "psycopg2 not available"
    
    try:
        database_url = os.getenv('DATABASE_URL', 'postgresql://postgres:2336@localhost:5432/pagila')
        conn = psycopg2.connect(database_url)
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT 1 as test")
            result = cursor.fetchone()
        conn.close()
        return True, "Database connection successful"
    except Exception as e:
        return False, f"Database connection failed: {str(e)}"

def execute_query(query: str, max_results: int = 100):
    """Execute a database query safely"""
    if not DB_AVAILABLE:
        return []
    
    try:
        database_url = os.getenv('DATABASE_URL', 'postgresql://postgres:2336@localhost:5432/pagila')
        conn = psycopg2.connect(database_url)
        conn.set_session(readonly=True)  # Read-only for safety
        
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Add LIMIT to prevent large result sets
            if "LIMIT" not in query.upper():
                query = f"{query.rstrip(';')} LIMIT {max_results}"
            
            cursor.execute(query)
            if cursor.description:
                results = cursor.fetchall()
                return [dict(row) for row in results]
            else:
                return []
    except Exception as e:
        st.error(f"Query execution error: {str(e)}")
        return []
    finally:
        if 'conn' in locals():
            conn.close()

def create_category_chart():
    """Create category distribution chart"""
    query = """
        SELECT c.name as category, COUNT(*) as count 
        FROM film f
        JOIN film_category fc ON f.film_id = fc.film_id
        JOIN category c ON fc.category_id = c.category_id
        GROUP BY c.name
        ORDER BY count DESC
        LIMIT 15
    """
    data = execute_query(query)
    if data:
        df = pd.DataFrame(data)
        fig = px.bar(
            df, 
            x='count', 
            y='category',
            orientation='h',
            title='📊 Film Categories Distribution',
            color='count',
            color_continuous_scale='viridis',
            labels={'count': 'Number of Films', 'category': 'Category'}
        )
        fig.update_layout(height=400)
        return fig
    return None

def create_rental_rate_chart():
    """Create rental rate distribution chart"""
    query = """
        SELECT 
            CASE 
                WHEN rental_rate < 1 THEN '< $1'
                WHEN rental_rate < 2 THEN '$1-2'
                WHEN rental_rate < 3 THEN '$2-3'
                WHEN rental_rate < 4 THEN '$3-4'
                ELSE '$4+'
            END as price_range,
            COUNT(*) as count
        FROM film
        GROUP BY price_range
        ORDER BY min(rental_rate)
    """
    data = execute_query(query)
    if data:
        df = pd.DataFrame(data)
        fig = px.pie(
            df, 
            names='price_range', 
            values='count',
            title='💰 Rental Rate Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        return fig
    return None

def create_length_histogram():
    """Create film length histogram"""
    query = "SELECT length FROM film WHERE length IS NOT NULL"
    data = execute_query(query)
    if data:
        df = pd.DataFrame(data)
        fig = px.histogram(
            df, 
            x='length',
            title='🎬 Film Length Distribution',
            labels={'length': 'Length (minutes)', 'count': 'Number of Films'},
            nbins=20
        )
        return fig
    return None

def get_database_metrics():
    """Get basic database metrics"""
    metrics = {}
    
    try:
        # Film count
        film_data = execute_query("SELECT COUNT(*) as count FROM film")
        metrics['film_count'] = film_data[0]['count'] if film_data else 0
        
        # Average rental rate
        rental_data = execute_query("SELECT ROUND(AVG(rental_rate), 2) as avg FROM film")
        metrics['avg_rental'] = rental_data[0]['avg'] if rental_data else 0
        
        # Longest film
        length_data = execute_query("SELECT MAX(length) as max_len FROM film")
        metrics['max_length'] = length_data[0]['max_len'] if length_data else 0
        
        # Active customers
        customer_data = execute_query("SELECT COUNT(*) as count FROM customer WHERE active = 1")
        metrics['active_customers'] = customer_data[0]['count'] if customer_data else 0
        
        # Total categories
        category_data = execute_query("SELECT COUNT(*) as count FROM category")
        metrics['total_categories'] = category_data[0]['count'] if category_data else 0
        
        # Total actors
        actor_data = execute_query("SELECT COUNT(*) as count FROM actor")
        metrics['total_actors'] = actor_data[0]['count'] if actor_data else 0
        
    except Exception as e:
        st.error(f"Metrics error: {str(e)}")
        metrics = {
            'film_count': 0,
            'avg_rental': 0,
            'max_length': 0,
            'active_customers': 0,
            'total_categories': 0,
            'total_actors': 0
        }
    
    return metrics

def render_sidebar():
    """Render sidebar with system information"""
    st.sidebar.title("🎬 Pagila AI Pro")
    
    # Database status
    db_status, db_message = test_database_connection()
    if db_status:
        st.sidebar.success("🗄️ Database: Connected")
    else:
        st.sidebar.error("🗄️ Database: Disconnected")
        st.sidebar.caption(db_message)
    
    # Quick stats
    if db_status:
        st.sidebar.subheader("📊 Quick Stats")
        metrics = get_database_metrics()
        st.sidebar.metric("Films", metrics['film_count'])
        st.sidebar.metric("Categories", metrics['total_categories'])
        st.sidebar.metric("Actors", metrics['total_actors'])
    
    # Session info
    st.sidebar.subheader("📈 Session Info")
    st.sidebar.metric("Queries Run", st.session_state.performance_stats['total_queries'])
    
    # Settings
    st.sidebar.subheader("⚙️ Settings")
    st.session_state.auto_visualization = st.sidebar.checkbox(
        "Auto Visualization", 
        value=st.session_state.get('auto_visualization', True)
    )
    st.session_state.max_results = st.sidebar.slider(
        "Max Results", 
        10, 500, 
        st.session_state.get('max_results', 100)
    )

def render_dashboard(db_connected):
    """Render dashboard with real data"""
    st.header("📊 Dashboard")
    
    if not db_connected:
        st.warning("⚠️ Dashboard requires database connection")
        st.info("Please check your database configuration in the Advanced Tools tab")
        return
    
    # Get and display metrics
    with st.spinner("Loading dashboard metrics..."):
        metrics = get_database_metrics()
    
    # Display metrics in columns
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("🎬 Films", f"{metrics['film_count']:,}")
    with col2:
        st.metric("💰 Avg Rental", f"${metrics['avg_rental']}")
    with col3:
        st.metric("⏱️ Max Length", f"{metrics['max_length']} min")
    with col4:
        st.metric("👥 Customers", f"{metrics['active_customers']:,}")
    with col5:
        st.metric("📂 Categories", f"{metrics['total_categories']}")
    with col6:
        st.metric("🎭 Actors", f"{metrics['total_actors']:,}")
    
    st.markdown("---")
    
    # Charts in rows
    st.subheader("📊 Data Visualizations")
    
    # First row: Category and Rental Rate charts
    col_left, col_right = st.columns(2)
    
    with col_left:
        with st.spinner("Creating category chart..."):
            fig_cat = create_category_chart()
            if fig_cat:
                st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st.error("Could not create category chart")
    
    with col_right:
        with st.spinner("Creating rental rate chart..."):
            fig_rental = create_rental_rate_chart()
            if fig_rental:
                st.plotly_chart(fig_rental, use_container_width=True)
            else:
                st.error("Could not create rental rate chart")
    
    # Second row: Length histogram
    st.subheader("🎬 Film Length Analysis")
    with st.spinner("Creating length histogram..."):
        fig_length = create_length_histogram()
        if fig_length:
            st.plotly_chart(fig_length, use_container_width=True)
        else:
            st.error("Could not create length histogram")

def render_query_interface(db_connected):
    """Render AI-powered natural language query interface"""
    st.header("🤖 AI Natural Language to SQL Interface")
    
    if not db_connected:
        st.warning("⚠️ Query interface requires database connection")
        return
    
    # AI Provider selection
    col_provider, col_model = st.columns([2, 2])
    
    with col_provider:
        ai_provider = st.selectbox(
            "🧠 Choose AI Provider:",
            options=["ollama", "openai", "gemini", "openrouter"],
            format_func=lambda x: {
                "ollama": "🦙 Ollama (Local & Free)",
                "openai": "🧠 OpenAI GPT",
                "gemini": "🔮 Google Gemini",
                "openrouter": "🔄 OpenRouter (Multiple Models)"
            }.get(x, x),
            index=0,
            help="Ollama is recommended for privacy and no API costs. OpenRouter provides access to many models including free ones."
        )
    
    with col_model:
        if ai_provider == "ollama":
            available_models = ["mistral:7b-instruct", "llama3.1:8b", "codellama:7b", "qwen2.5-coder:7b"]
            current_model = os.getenv('OLLAMA_MODEL', 'mistral:7b-instruct')
            model = st.selectbox("Model:", available_models, 
                               index=available_models.index(current_model) if current_model in available_models else 0)
        elif ai_provider == "openai":
            openai_models = [
                "gpt-4o-mini",
                "gpt-4o",
                "gpt-4-turbo",
                "gpt-4",
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k"
            ]
            current_model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
            model = st.selectbox("Model:", openai_models,
                               index=openai_models.index(current_model) if current_model in openai_models else 0,
                               help="gpt-4o-mini is the most cost-effective option")
        elif ai_provider == "gemini":
            gemini_models = [
                "gemini-1.5-pro",
                "gemini-1.5-flash",
                "gemini-pro",
                "gemini-pro-vision"
            ]
            current_model = os.getenv('GEMINI_MODEL', 'gemini-pro')
            model = st.selectbox("Model:", gemini_models,
                               index=gemini_models.index(current_model) if current_model in gemini_models else 0,
                               help="gemini-1.5-flash is faster, gemini-1.5-pro is more capable")
        elif ai_provider == "openrouter":
            openrouter_models = [
                "deepseek/deepseek-chat-v3-0324:free",
                "meta-llama/llama-3.1-8b-instruct:free",
                "microsoft/phi-3-mini-128k-instruct:free", 
                "google/gemma-2-9b-it:free",
                "mistralai/mistral-7b-instruct:free",
                "meta-llama/llama-3.1-70b-instruct",
                "anthropic/claude-3.5-sonnet",
                "openai/gpt-4o-mini"
            ]
            current_model = os.getenv('OPENROUTER_MODEL', 'deepseek/deepseek-chat-v3-0324:free')
            model = st.selectbox("Model:", openrouter_models,
                               index=openrouter_models.index(current_model) if current_model in openrouter_models else 0,
                               help="Models marked ':free' are free to use!")
        else:
            st.info(f"Using {ai_provider} default model")
    
    st.markdown("---")
    
    # Example questions
    st.subheader("💡 Try these example questions:")
    
    example_questions = [
        "How many movies are in the database?",
        "What are the top 5 longest movies?",
        "Which actors have appeared in the most films?",
        "What are the most popular film categories?",
        "Show me all movies with a rental rate above $4",
        "Which customers have rented the most movies?",
        "What is the average rental rate by category?",
        "Find all movies released after 2005",
        "Show me the top 10 highest earning films",
        "List all comedy movies with their ratings"
    ]
    
    # Display examples in a nice grid
    cols = st.columns(2)
    for i, question in enumerate(example_questions):
        col = cols[i % 2]
        with col:
            if st.button(f"� {question}", key=f"example_{i}", use_container_width=True):
                st.session_state.nl_question = question
    
    st.markdown("---")
    
    # Natural Language Question Input
    st.subheader("🗣️ Ask your question in natural language:")
    
    user_question = st.text_area(
        "Type your question here:",
        value=st.session_state.get('nl_question', ''),
        height=100,
        placeholder="Example: How many action movies are there? What are the top rated films?",
        help="Ask any question about the Pagila movie database in plain English!"
    )
    
    # Store question in session state
    st.session_state.nl_question = user_question
    
    # Action buttons
    col_convert, col_execute, col_clear = st.columns([2, 2, 1])
    
    with col_convert:
        convert_button = st.button("� Convert to SQL", type="primary", use_container_width=True)
    
    with col_execute:
        execute_button = st.button("� Convert & Execute", type="secondary", use_container_width=True)
    
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.nl_question = ""
            st.session_state.generated_sql = ""
            st.rerun()
    
    # Convert natural language to SQL
    if convert_button or execute_button:
        if not user_question.strip():
            st.warning("⚠️ Please enter a question first.")
            return
        
        with st.spinner(f"🤖 Converting your question to SQL using {ai_provider.title()}... This may take up to 2 minutes for complex questions."):
            start_time = time.time()
            
            # Call AI to convert natural language to SQL (model is already selected above)
            result = natural_language_to_sql(user_question, ai_provider, model)
            
            conversion_time = time.time() - start_time
        
        if result.get('error'):
            st.error(f"❌ AI Conversion Error: {result['error']}")
            if result.get('raw_response'):
                with st.expander("🔍 Raw AI Response"):
                    st.text(result['raw_response'])
        
        elif result.get('sql'):
            generated_sql = result['sql']
            st.session_state.generated_sql = generated_sql
            
            st.success(f"✅ Question converted to SQL in {conversion_time:.2f} seconds")
            
            # Display the generated SQL
            st.subheader("🔍 Generated SQL Query:")
            st.code(generated_sql, language='sql')
            
            # Show AI reasoning if available
            if result.get('raw_response') and result['raw_response'] != generated_sql:
                with st.expander("🧠 AI Reasoning (Click to expand)"):
                    st.text(result['raw_response'])
            
            # Execute the query if requested
            if execute_button:
                st.subheader("📊 Query Results:")
                
                with st.spinner("Executing generated SQL query..."):
                    exec_start_time = time.time()
                    query_results = execute_query(generated_sql, st.session_state.max_results)
                    exec_time = time.time() - exec_start_time
                
                if query_results:
                    st.success(f"✅ Query executed in {exec_time:.3f} seconds")
                    st.info(f"📊 Found {len(query_results)} results")
                    
                    # Display results as DataFrame
                    df = pd.DataFrame(query_results)
                    st.dataframe(df, use_container_width=True)
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"ai_query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Add to history
                    st.session_state.query_history.append({
                        'question': user_question,
                        'sql': generated_sql,
                        'timestamp': datetime.now(),
                        'duration': exec_time,
                        'conversion_time': conversion_time,
                        'row_count': len(query_results),
                        'provider': ai_provider
                    })
                    st.session_state.performance_stats['total_queries'] += 1
                    
                    # Auto-visualization
                    if st.session_state.auto_visualization and len(df) > 1:
                        render_auto_visualization(df, user_question)
                
                else:
                    st.info("ℹ️ Query executed successfully but returned no results")
        
        else:
            st.error("❌ Could not generate SQL from your question. Please try rephrasing.")
    
    # Manual SQL execution section
    st.markdown("---")
    st.subheader("⚙️ Manual SQL Editor")
    
    manual_sql = st.text_area(
        "Edit or write SQL directly:",
        value=st.session_state.get('generated_sql', ''),
        height=120,
        placeholder="SELECT * FROM film LIMIT 10;",
        help="You can edit the generated SQL or write your own"
    )
    
    if st.button("▶️ Execute Manual SQL", use_container_width=True):
        if manual_sql.strip():
            with st.spinner("Executing SQL..."):
                start_time = time.time()
                results = execute_query(manual_sql, st.session_state.max_results)
                exec_time = time.time() - start_time
            
            if results:
                st.success(f"✅ Manual query executed in {exec_time:.3f} seconds")
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                
                # Add to history
                st.session_state.query_history.append({
                    'question': 'Manual SQL',
                    'sql': manual_sql,
                    'timestamp': datetime.now(),
                    'duration': exec_time,
                    'conversion_time': 0,
                    'row_count': len(results),
                    'provider': 'manual'
                })
            else:
                st.info("Query executed but returned no results")
        else:
            st.warning("Please enter a SQL query")
    
    # Query history with AI context
    if st.session_state.get('query_history', []):
        st.markdown("---")
        st.subheader("📜 AI Query History")
        
        # Show only last 5 queries
        recent_queries = list(reversed(st.session_state.query_history[-5:]))
        
        for i, entry in enumerate(recent_queries, 1):
            provider_emoji = {
                'ollama': '🦙',
                'openai': '🧠', 
                'gemini': '🔮',
                'openrouter': '🔄',
                'manual': '⚙️'
            }.get(entry.get('provider', 'manual'), '❓')
            
            timestamp = entry['timestamp'].strftime('%H:%M:%S')
            duration = entry['duration']
            conversion_time = entry.get('conversion_time', 0)
            
            with st.expander(f"{provider_emoji} {i}. {timestamp} - Exec: {duration:.3f}s | Conv: {conversion_time:.2f}s | Rows: {entry['row_count']}"):
                st.write(f"**Question:** {entry['question']}")
                st.code(entry['sql'], language='sql')
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"🔄 Re-run Question", key=f"rerun_q_{i}"):
                        st.session_state.nl_question = entry['question']
                        st.rerun()
                
                with col2:
                    if st.button(f"📝 Edit SQL", key=f"edit_sql_{i}"):
                        st.session_state.generated_sql = entry['sql']
                        st.rerun()

def render_auto_visualization(df: pd.DataFrame, question: str):
    """Render automatic visualizations based on query results"""
    st.subheader("📊 Automatic Visualization")
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if len(numeric_cols) == 0:
        st.info("No numeric data available for visualization")
        return
    
    # Determine best visualization based on data
    if len(df) == 1:
        # Single row - show metrics
        st.subheader("� Key Metrics")
        cols = st.columns(min(len(numeric_cols), 4))
        for i, col in enumerate(numeric_cols[:4]):
            with cols[i]:
                st.metric(col.replace('_', ' ').title(), df[col].iloc[0])
    
    elif len(categorical_cols) > 0 and len(numeric_cols) > 0:
        # Categorical + Numeric - Bar chart
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        
        # Limit to top 20 categories for readability
        if len(df) > 20:
            df_plot = df.nlargest(20, num_col)
        else:
            df_plot = df
        
        fig = px.bar(
            df_plot, 
            x=cat_col, 
            y=num_col,
            title=f"{num_col.replace('_', ' ').title()} by {cat_col.replace('_', ' ').title()}",
            color=num_col,
            color_continuous_scale='viridis'
        )
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    elif len(numeric_cols) == 1:
        # Single numeric column - histogram
        num_col = numeric_cols[0]
        fig = px.histogram(
            df, 
            x=num_col, 
            title=f"Distribution of {num_col.replace('_', ' ').title()}",
            nbins=min(20, len(df)//2) if len(df) > 10 else 10
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif len(numeric_cols) >= 2:
        # Multiple numeric columns - scatter plot
        x_col, y_col = numeric_cols[0], numeric_cols[1]
        
        # Add color if there's a categorical column
        color_col = categorical_cols[0] if categorical_cols else None
        
        fig = px.scatter(
            df.head(100),  # Limit points for performance
            x=x_col, 
            y=y_col,
            color=color_col,
            title=f"{x_col.replace('_', ' ').title()} vs {y_col.replace('_', ' ').title()}",
            hover_data=numeric_cols[:3]  # Show top 3 numeric columns on hover
        )
        st.plotly_chart(fig, use_container_width=True)

def render_advanced_tools():
    """Render advanced tools and settings"""
    st.header("🔧 Advanced Tools")
    
    # Database connection section
    st.subheader("🗄️ Database Configuration")
    
    # Current database info
    database_url = os.getenv('DATABASE_URL', 'postgresql://postgres:2336@localhost:5432/pagila')
    
    # Parse URL for display
    try:
        import urllib.parse
        parsed = urllib.parse.urlparse(database_url)
        current_config = {
            'host': parsed.hostname or 'localhost',
            'port': parsed.port or 5432,
            'database': parsed.path.lstrip('/') or 'pagila',
            'username': parsed.username or 'postgres'
        }
    except:
        current_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'pagila',
            'username': 'postgres'
        }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input("Host", value=current_config['host'], disabled=True)
        st.text_input("Database", value=current_config['database'], disabled=True)
    
    with col2:
        st.text_input("Port", value=str(current_config['port']), disabled=True)
        st.text_input("Username", value=current_config['username'], disabled=True)
    
    # Connection test
    if st.button("🔍 Test Database Connection"):
        with st.spinner("Testing connection..."):
            db_status, db_message = test_database_connection()
            if db_status:
                st.success(f"✅ {db_message}")
                
                # Show database info
                try:
                    info_query = """
                    SELECT 
                        version() as version,
                        current_database() as database,
                        current_user as user,
                        now() as current_time
                    """
                    info = execute_query(info_query)
                    if info:
                        st.info(f"Database: {info[0]['database']} | User: {info[0]['user']}")
                        st.caption(f"Version: {info[0]['version'][:50]}...")
                except:
                    pass
            else:
                st.error(f"❌ {db_message}")
    
    st.markdown("---")
    
    # AI Provider settings
    st.subheader("🤖 AI Provider Configuration")
    
    current_provider = os.getenv('AI_PROVIDER', 'ollama')
    
    provider_options = {
        'ollama': '🦙 Ollama (Local)',
        'openai': '🧠 OpenAI',
        'gemini': '🔮 Google Gemini',
        'openrouter': '🔄 OpenRouter'
    }
    
    selected_provider = st.selectbox(
        "Choose AI Provider:",
        options=list(provider_options.keys()),
        format_func=lambda x: provider_options[x],
        index=list(provider_options.keys()).index(current_provider) if current_provider in provider_options else 0
    )
    
    if selected_provider == 'ollama':
        st.info("🦙 **Ollama (Recommended)** - Free local AI, no API key required!")
        
        ollama_host = st.text_input("Ollama Host", value=os.getenv('OLLAMA_HOST', 'http://localhost:11434'))
        ollama_model = st.text_input("Model", value=os.getenv('OLLAMA_MODEL', 'mistral:7b-instruct'))
        
        if st.button("Test Ollama Connection"):
            try:
                import requests
                response = requests.get(f"{ollama_host}/api/tags", timeout=10)
                if response.status_code == 200:
                    st.success("✅ Ollama is running!")
                    models = response.json().get('models', [])
                    if models:
                        model_names = [m['name'] for m in models]
                        st.success(f"📋 Available models: {', '.join(model_names[:5])}")
                        if len(model_names) > 5:
                            st.caption(f"...and {len(model_names) - 5} more")
                    else:
                        st.warning("No models found. Pull a model with: ollama pull mistral")
                else:
                    st.error(f"❌ Connection failed: HTTP {response.status_code}")
            except requests.exceptions.Timeout:
                st.error("❌ Connection timed out. Make sure Ollama is running and accessible.")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to Ollama. Check if the service is running.")
            except Exception as e:
                st.error(f"❌ Cannot connect to Ollama: {str(e)}")
                st.info("Make sure Ollama is installed and running. Visit: https://ollama.ai")
    
    elif selected_provider == 'openai':
        api_key = os.getenv('OPENAI_API_KEY', '')
        masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "Not set"
        st.text_input("API Key", value=masked_key, disabled=True, type="password")
        st.selectbox("Model", ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"], 
                    index=0 if os.getenv('OPENAI_MODEL') == 'gpt-4o-mini' else 0)
    
    elif selected_provider == 'gemini':
        api_key = os.getenv('GEMINI_API_KEY', '')
        masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "Not set"
        st.text_input("API Key", value=masked_key, disabled=True, type="password")
        st.selectbox("Model", ["gemini-pro", "gemini-pro-vision"], index=0)
    
    elif selected_provider == 'openrouter':
        st.info("🔄 **OpenRouter** - Access to multiple AI models including free ones!")
        
        api_key = os.getenv('OPENROUTER_API_KEY', '')
        masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "Not set"
        st.text_input("API Key", value=masked_key, disabled=True, type="password")
        
        current_model = os.getenv('OPENROUTER_MODEL', 'deepseek/deepseek-chat-v3-0324:free')
        st.text_input("Model", value=current_model, disabled=True)
        
        if ":free" in current_model:
            st.success("✅ Using a FREE model!")
        else:
            st.info("💰 This model requires credits")
        
        if st.button("Test OpenRouter Connection"):
            if api_key:
                try:
                    test_prompt = "Convert this to SQL: How many records are there?"
                    response = call_openrouter_api(test_prompt)
                    if "Error:" not in response:
                        st.success("✅ OpenRouter connection successful!")
                        st.info(f"Model response preview: {response[:100]}...")
                    else:
                        st.error(f"❌ {response}")
                except Exception as e:
                    st.error(f"❌ Connection test failed: {str(e)}")
            else:
                st.error("❌ No API key configured")
    
    st.markdown("---")
    
    # System information
    st.subheader("ℹ️ System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Python Version", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        st.metric("Streamlit Version", st.__version__)
        st.metric("Current Provider", selected_provider.title())
    
    with col2:
        st.metric("Total Queries", st.session_state.performance_stats['total_queries'])
        st.metric("Database Status", "Connected" if test_database_connection()[0] else "Disconnected")
        st.metric("Dependencies", f"{'✅' if DB_AVAILABLE else '❌'} DB | {'✅' if DOTENV_AVAILABLE else '❌'} ENV")

def main():
    """Main application function"""
    # Page configuration
    st.set_page_config(
        page_title="🎬 Pagila AI Assistant Pro",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Initialize session state
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    
    if 'performance_stats' not in st.session_state:
        st.session_state.performance_stats = {
            'total_queries': 0,
            'avg_response_time': 0,
            'success_rate': 100
        }
    
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    
    if 'current_provider' not in st.session_state:
        st.session_state.current_provider = 'ollama'
    
    if 'auto_visualization' not in st.session_state:
        st.session_state.auto_visualization = True
    
    if 'max_results' not in st.session_state:
        st.session_state.max_results = 100
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    st.title("🎬 Pagila AI Assistant Pro")
    st.markdown("*AI-powered database querying and analysis system*")
    
    # Test database connection for main content
    db_status, db_message = test_database_connection()
    
    # Status indicator
    if db_status:
        st.success(f"✅ {db_message}")
    else:
        st.error(f"❌ {db_message}")
        st.info("💡 **Tip**: Make sure PostgreSQL is running and check your connection settings in the Advanced Tools tab")
    
    # Navigation tabs
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 AI Query Assistant", "🔧 Advanced Tools"])
    
    with tab1:
        render_dashboard(db_status)
    
    with tab2:
        render_query_interface(db_status)
    
    with tab3:
        render_advanced_tools()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.9em; padding: 20px;'>"
        "🦙 <strong>Ollama Integration</strong> - Free Local AI | "
        "🎬 <strong>Pagila Database</strong> - Sample Cinema Data | "
        "Built with <strong>Streamlit</strong> and ❤️"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
