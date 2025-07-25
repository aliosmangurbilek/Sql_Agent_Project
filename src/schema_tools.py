import psycopg2
from psycopg2.extras import RealDictCursor
from collections import defaultdict
from typing import Dict, List, Any
import sqlglot
import sqlglot.expressions as exp
import asyncio
import os
import json
import aiohttp
from openai import AsyncOpenAI
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
# Try to load from config directory first, then fall back to current directory
env_path = Path(__file__).parent.parent / "config" / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ schema_tools: Loaded environment from {env_path}")
else:
    load_dotenv()
    print("✅ schema_tools: Loaded environment from current directory")


async def get_embedding(text: str) -> list:
    """Generate embedding for text using Ollama."""
    
    try:
        base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        model = os.getenv('OLLAMA_EMBEDDING_MODEL', 'mxbai-embed-large')
        
        print(f"🔗 Connecting to {base_url} with model {model}")
        
        async with aiohttp.ClientSession() as session:
            embed_url = f"{base_url}/api/embeddings"
            payload = {
                "model": model,
                "prompt": text
            }
            
            async with session.post(embed_url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    embedding = result.get('embedding', [])
                    print(f"✅ Generated embedding with {len(embedding)} dimensions")
                    return embedding
                else:
                    print(f"❌ Embedding API error: {response.status}")
                    return []
                    
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return []


class SQLValidationError(Exception):
    """Base exception for SQL validation errors."""
    pass


class UnsafeSQLError(SQLValidationError):
    """Raised when SQL contains unsafe operations."""
    pass


class InvalidSQLError(SQLValidationError):
    """Raised when SQL cannot be parsed or is invalid."""
    pass


class MultipleStatementsError(SQLValidationError):
    """Raised when SQL contains multiple statements."""
    pass


def verify_sql(query: str) -> str | None:
    """
    Verifies and sanitizes a SQL query to ensure it's safe for execution.
    
    Args:
        query: The SQL query string to verify.
        
    Returns:
        The safe SQL string with LIMIT added if needed, or None if unsafe.
        
    Raises:
        InvalidSQLError: If the SQL cannot be parsed or is invalid.
        UnsafeSQLError: If the SQL contains unsafe operations (INSERT, UPDATE, DELETE, DROP).
        MultipleStatementsError: If the SQL contains multiple statements.
    """
    if not query or not query.strip():
        raise InvalidSQLError("Query cannot be empty")
    
    try:
        # Parse the SQL with PostgreSQL dialect
        parsed = sqlglot.parse(query.strip(), dialect="postgres")
    except Exception as e:
        raise InvalidSQLError(f"Failed to parse SQL: {str(e)}")
    
    # Ensure only one statement is present
    if len(parsed) != 1:
        if len(parsed) == 0:
            raise InvalidSQLError("No valid SQL statements found")
        else:
            raise MultipleStatementsError(f"Multiple statements found ({len(parsed)}), only single SELECT statements are allowed")
    
    statement = parsed[0]
    
    # Check for unsafe operations
    unsafe_expressions = (exp.Insert, exp.Delete, exp.Update, exp.Drop)
    if statement.find(unsafe_expressions):
        unsafe_types = []
        for unsafe_type in unsafe_expressions:
            if statement.find(unsafe_type):
                unsafe_types.append(unsafe_type.__name__.upper())
        raise UnsafeSQLError(f"Unsafe operations detected: {', '.join(unsafe_types)}")
    
    # Ensure it's a SELECT statement
    if not isinstance(statement, exp.Select):
        raise UnsafeSQLError(f"Only SELECT statements are allowed, got {type(statement).__name__}")
    
    # Check if LIMIT is already present
    if not statement.find(exp.Limit):
        # Add LIMIT 1000 if not present
        statement = statement.limit(1000)
    
    # Return the safe SQL string
    return statement.sql(dialect="postgres")


def extract_schema(conn) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extracts the database schema for tables in the 'public' schema.

    Args:
        conn: A psycopg2 connection object.

    Returns:
        A dictionary representing the schema, with table names as keys
        and a list of column dictionaries as values.
    """
    schema = defaultdict(list)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            query = """
                SELECT
                    table_name,
                    column_name,
                    data_type
                FROM
                    information_schema.columns
                WHERE
                    table_schema = 'public'
                ORDER BY
                    table_name,
                    ordinal_position;
            """
            cursor.execute(query)
            rows = cursor.fetchall()

            for row in rows:
                schema[row['table_name']].append({
                    "column": row['column_name'],
                    "type": row['data_type']
                })

    except psycopg2.Error as e:
        print(f"Database error: {e}")
        # Depending on requirements, you might want to re-raise the exception
        # or handle it in a different way.
        # raise e
        return {} # Return an empty dict on error

    # The defaultdict is converted to a regular dict by most JSON serializers,
    # but it's good practice to convert it explicitly if the function signature
    # promises a dict.
    return dict(sorted(schema.items()))

def build_prompt(question: str, schema_dict: dict) -> tuple[list, list]:
    """
    Builds a prompt for an OpenAI chat model to generate a SQL query.

    Args:
        question: The user's question to be answered with a SQL query.
        schema_dict: A dictionary representing the database schema.

    Returns:
        A tuple containing the messages list for the chat model and
        the functions specification.
    """
    schema_lines = ["### Database Schema ###"]
    for table, columns in schema_dict.items():
        column_defs = ", ".join([f"{c['column']} {c['type']}" for c in columns])
        schema_lines.append(f"Table {table} ({column_defs})")
    
    # Add relationship information for better understanding
    schema_lines.append("\n### Important Relationships ###")
    schema_lines.append("- film_category.film_id → film.film_id")
    schema_lines.append("- film_category.category_id → category.category_id")
    schema_lines.append("- film_actor.film_id → film.film_id")
    schema_lines.append("- film_actor.actor_id → actor.actor_id")
    schema_lines.append("- rental.customer_id → customer.customer_id")
    schema_lines.append("- rental.inventory_id → inventory.inventory_id")
    schema_lines.append("- inventory.film_id → film.film_id")
    schema_lines.append("- inventory.store_id → store.store_id")
    
    # Add common query patterns
    schema_lines.append("\n### Common Query Examples ###")
    schema_lines.append("-- For film categories by name:")
    schema_lines.append("-- SELECT c.name, COUNT(*) FROM film f")
    schema_lines.append("-- JOIN film_category fc ON f.film_id = fc.film_id")
    schema_lines.append("-- JOIN category c ON fc.category_id = c.category_id")
    schema_lines.append("-- GROUP BY c.name")
    
    schema_text = "\n".join(schema_lines)

    prompt = f"""{schema_text}

{question}

IMPORTANT RULES:
1. ALWAYS use proper JOINs when referencing related tables
2. For category names, use: film → film_category → category (JOIN on IDs)
3. For actor names, use: film → film_actor → actor (JOIN on IDs)
4. film_category table only has film_id and category_id, NOT category name
5. category table has category_id and name columns
6. Use table aliases for cleaner queries

You must use the generate_sql function to provide a PostgreSQL SELECT query that answers the question. 
The query parameter must contain only the SQL query string.
Example: {{"query": "SELECT c.name, COUNT(*) FROM film f JOIN film_category fc ON f.film_id = fc.film_id JOIN category c ON fc.category_id = c.category_id GROUP BY c.name ORDER BY COUNT(*) DESC"}}

Use only the given schema tables. Generate a valid PostgreSQL SELECT query with proper JOINs."""

    messages = [{"role": "user", "content": prompt}]

    functions = [
        {
            "type": "function",
            "function": {
                "name": "generate_sql",
                "description": "Generates a PostgreSQL SELECT query to answer the user's question",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "A valid PostgreSQL SELECT query"
                        }
                    },
                    "required": ["query"],
                },
            }
        }
    ]
    return messages, functions


async def ask_db(question: str) -> list[dict]:
    """
    Asks the database a question by generating SQL via OpenAI GPT and executing it safely.
    
    This function uses OpenAI's GPT models with function calling for more reliable SQL generation.
    
    Environment Variables:
        DATABASE_URL (required): PostgreSQL connection string
        OPENAI_API_KEY (required): OpenAI API key
        OPENAI_MODEL (optional): Model to use (default: gpt-3.5-turbo)
        USE_OLLAMA (optional): Set to 'true' to use Ollama instead of OpenAI
    
    Args:
        question: The natural language question to ask the database.
        
    Returns:
        A list of dictionaries representing the query results.
        
    Raises:
        ValueError: If no database URL or API key is configured.
        SQLValidationError: If the generated SQL is unsafe or invalid.
        psycopg2.Error: If database connection or execution fails.
        Exception: If OpenAI API call fails or returns unexpected format.
        
    Example:
        # Set up environment
        export DATABASE_URL='postgresql://postgres:2336@localhost:5432/pagila'
        export OPENAI_API_KEY='your-openai-api-key'
        export OPENAI_MODEL='gpt-4'  # optional, defaults to gpt-3.5-turbo
        
        # Use the function
        result = await ask_db("How many films are in the database?")
    """
    # Check for required environment variables
    database_url = os.getenv('DATABASE_URL')
    ai_provider = os.getenv('AI_PROVIDER', 'ollama').lower()
    
    # Provider-specific configurations
    openai_api_key = os.getenv('OPENAI_API_KEY')
    openai_model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    gemini_model = os.getenv('GEMINI_MODEL', 'gemini-pro')
    
    openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
    openrouter_model = os.getenv('OPENROUTER_MODEL', 'meta-llama/llama-3.1-8b-instruct:free')
    openrouter_site_url = os.getenv('OPENROUTER_SITE_URL', 'http://localhost:8501')
    openrouter_app_name = os.getenv('OPENROUTER_APP_NAME', 'IGA_Staj_Project')
    
    # Legacy support for USE_OLLAMA
    use_ollama = os.getenv('USE_OLLAMA', 'false').lower() == 'true'
    if use_ollama and ai_provider == 'ollama':
        ai_provider = 'ollama'
    
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is required")
    
    # Step 1: Get database schema first (using asyncio.to_thread for blocking DB work)
    def get_schema():
        try:
            # Connect to get schema - this connection is only for schema extraction
            conn = psycopg2.connect(database_url)
            schema_dict = extract_schema(conn)
            conn.close()
            return schema_dict
        except psycopg2.Error as e:
            raise psycopg2.Error(f"Failed to extract database schema: {e}")
    
    schema_dict = await asyncio.to_thread(get_schema)
    
    if not schema_dict:
        raise ValueError("Could not extract database schema or schema is empty")
    
    # Step 2: Build prompt and functions via build_prompt
    messages, functions = build_prompt(question, schema_dict)
    
    # Step 3: Choose API based on AI_PROVIDER environment variable
    if ai_provider == 'openai':
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required when AI_PROVIDER=openai")
        generated_query = await _call_openai_api(messages, functions, openai_model, openai_api_key)
    
    elif ai_provider == 'gemini':
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required when AI_PROVIDER=gemini")
        generated_query = await _call_gemini_api(messages, functions, gemini_model, gemini_api_key)
    
    elif ai_provider == 'openrouter':
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required when AI_PROVIDER=openrouter")
        generated_query = await _call_openrouter_api(messages, functions, openrouter_model, openrouter_api_key, openrouter_site_url, openrouter_app_name)
    
    elif ai_provider == 'ollama':
        generated_query = await _call_ollama_api(messages, functions)
    
    else:
        raise ValueError(f"Unsupported AI_PROVIDER: {ai_provider}. Supported providers: openai, gemini, openrouter, ollama")
    
    # Debug: Print the generated SQL query
    print(f"🔍 Generated SQL Query: {generated_query}")
    
    # Step 4: Pass through verify_sql
    try:
        safe_query = verify_sql(generated_query)
    except SQLValidationError:
        # Re-raise validation errors as-is
        raise
    
    if safe_query is None:
        raise UnsafeSQLError("Generated query was deemed unsafe by verify_sql")
    
    # Step 5: Execute SQL with READ-ONLY connection and statement_timeout
    def execute_query():
        try:
            # Parse the DATABASE_URL to add read-only parameters
            conn_params = psycopg2.extensions.parse_dsn(database_url)
            
            # Add read-only and timeout settings
            conn_params.update({
                'options': '-c default_transaction_read_only=on -c statement_timeout=3000'
            })
            
            # Connect with read-only settings
            conn = psycopg2.connect(**conn_params)
            
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(safe_query)
                    rows = cursor.fetchall()
                    
                    # Convert RealDictRow objects to regular dictionaries
                    return [dict(row) for row in rows]
            finally:
                conn.close()
                
        except psycopg2.Error as e:
            raise psycopg2.Error(f"Database execution failed: {e}")
    
    # Use asyncio.to_thread for blocking database work
    result_rows = await asyncio.to_thread(execute_query)
    
    return result_rows


async def _call_openai_api(messages: list, functions: list, model: str, api_key: str) -> str:
    """Call OpenAI API for SQL generation."""
    try:
        client = AsyncOpenAI(api_key=api_key)
        
        # Convert our function format to OpenAI's tools format
        tools = [
            {
                "type": "function",
                "function": func["function"]
            } for func in functions
        ]
        
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.0
        )
        
        # Extract the function call
        message = response.choices[0].message
        
        if not message.tool_calls:
            raise Exception("OpenAI did not return any tool calls")
        
        tool_call = message.tool_calls[0]
        function_args = json.loads(tool_call.function.arguments)
        
        # Extract the query
        generated_query = (
            function_args.get('query') or 
            function_args.get('sql') or 
            function_args.get('sql_query') or
            function_args.get('statement')
        )
        
        if not generated_query:
            raise Exception("OpenAI did not return a query in the function arguments")
        
        return generated_query
        
    except Exception as e:
        raise Exception(f"OpenAI API call failed: {e}")


async def _call_gemini_api(messages: list, functions: list, model: str, api_key: str) -> str:
    """Call Google Gemini API for SQL generation."""
    try:
        genai.configure(api_key=api_key)
        
        # Convert messages to Gemini format
        conversation_text = ""
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_text += f"{role}: {msg['content']}\n"
        
        # Create a simpler prompt that asks for just the SQL query
        prompt = f"""{conversation_text}

Please respond with ONLY a valid PostgreSQL SELECT query to answer the question. 
Do not include any JSON, markdown, or explanations. Just return the SQL query directly.

Example: SELECT COUNT(*) FROM table_name;"""
        
        # Create model instance
        model_instance = genai.GenerativeModel(model)
        
        # Generate response
        response = model_instance.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean up the response - remove any markdown or extra formatting
        import re
        
        # Remove markdown code blocks
        response_text = re.sub(r'```sql\n?', '', response_text)
        response_text = re.sub(r'```\n?', '', response_text)
        
        # Extract SQL query (look for SELECT statements)
        sql_match = re.search(r'(SELECT[^;]*;?)', response_text, re.IGNORECASE | re.DOTALL)
        if sql_match:
            generated_query = sql_match.group(1).strip()
        else:
            # Fallback: use the entire cleaned response
            generated_query = response_text.strip()
        
        # Remove trailing semicolon if present
        if generated_query.endswith(';'):
            generated_query = generated_query[:-1]
        
        if not generated_query or not generated_query.upper().startswith('SELECT'):
            raise Exception(f"Gemini did not return a valid SQL query: {generated_query}")
        
        return generated_query
        
    except Exception as e:
        raise Exception(f"Gemini API call failed: {e}")


async def _call_ollama_api(messages: list, functions: list) -> str:
    """Call Ollama API for SQL generation (fallback)."""
    ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    ollama_model = os.getenv('OLLAMA_MODEL', 'mistral:7b-instruct')
    
    # First, try to check if Ollama is running
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{ollama_host}/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status != 200:
                    raise Exception(f"Ollama not responding at {ollama_host}")
    except Exception as e:
        raise Exception(f"Cannot connect to Ollama at {ollama_host}. Please ensure Ollama is running: {e}")
    
    ollama_payload = {
        "model": ollama_model,
        "messages": messages,
        "tools": functions,
        "options": {
            "temperature": 0.0
        },
        "stream": False
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{ollama_host}/api/chat",
                json=ollama_payload,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ollama API returned {response.status}: {error_text}")
                
                response_data = await response.json()
                
    except aiohttp.ClientError as e:
        raise Exception(f"Failed to connect to Ollama at {ollama_host}: {e}")
    except Exception as e:
        raise Exception(f"Ollama API call failed: {e}")
    
    # Extract the query from tool_calls
    message = response_data.get('message', {})
    tool_calls = message.get('tool_calls', [])
    
    if not tool_calls:
        # Fallback: try to extract SQL from the message content
        content = message.get('content', '')
        if content:
            import re
            # Look for SQL in the content - more robust extraction
            sql_patterns = [
                r'```sql\s*(.*?)\s*```',  # SQL in code blocks
                r'```\s*(SELECT[^`]*)\s*```',  # SELECT in code blocks
                r'(SELECT[^.]*?)(?:\s*\.\s*|\s*$)',  # SELECT until period or end
                r'(SELECT[^;]*;?)',  # Basic SELECT pattern
            ]
            
            for pattern in sql_patterns:
                sql_match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
                if sql_match:
                    extracted_sql = sql_match.group(1).strip()
                    # Clean up common extra text
                    extracted_sql = re.sub(r'\s*\.\s*This.*$', '', extracted_sql, flags=re.IGNORECASE)
                    extracted_sql = re.sub(r'\s*,\s*which.*$', '', extracted_sql, flags=re.IGNORECASE)
                    extracted_sql = extracted_sql.rstrip(';.,')
                    return extracted_sql
        raise Exception("Ollama did not return any tool calls or valid SQL")
    
    first_tool_call = tool_calls[0]
    
    try:
        tool_function = first_tool_call.get('function', {})
        function_name = tool_function.get('name', '')
        
        if not function_name or function_name.lower() in ['', 'none']:
            raise Exception(f"Ollama returned invalid function name: {function_name}")
        
        arguments = tool_function.get('arguments', {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        
        generated_query = (
            arguments.get('query') or 
            arguments.get('sql') or 
            arguments.get('sql_query') or
            arguments.get('statement')
        )
            
    except (KeyError, TypeError, json.JSONDecodeError) as e:
        raise Exception(f"Failed to parse Ollama tool arguments: {e}")
    
    if not generated_query:
        raise Exception("Ollama did not return a query in the tool arguments")
    
    return generated_query


async def _call_openrouter_api(messages: list, functions: list, model: str, api_key: str, site_url: str, app_name: str) -> str:
    """Call OpenRouter API for SQL generation."""
    try:
        # OpenRouter uses OpenAI-compatible API format
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": site_url,
            "X-Title": app_name,
            "Content-Type": "application/json"
        }
        
        # Convert our function format to OpenAI's tools format
        tools = [
            {
                "type": "function",
                "function": func["function"]
            } for func in functions
        ]
        
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.0
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"OpenRouter API returned {response.status}: {error_text}")
                
                response_data = await response.json()
        
        # Extract the function call (same format as OpenAI)
        message = response_data["choices"][0]["message"]
        
        if not message.get("tool_calls"):
            raise Exception("OpenRouter did not return any tool calls")
        
        tool_call = message["tool_calls"][0]
        function_args = json.loads(tool_call["function"]["arguments"])
        
        # Extract the query
        generated_query = (
            function_args.get('query') or 
            function_args.get('sql') or 
            function_args.get('sql_query') or
            function_args.get('statement')
        )
        
        if not generated_query:
            raise Exception("OpenRouter did not return a query in the function arguments")
        
        return generated_query
        
    except Exception as e:
        raise Exception(f"OpenRouter API call failed: {e}")


async def search_documents(query: str, top_n: int = 3) -> list[dict]:
    """
    Search for similar documents using vector similarity (pgvector) with text search fallback.
    
    This function first tries to use vector similarity search with embeddings.
    If that fails or no embeddings are available, it falls back to PostgreSQL full-text search.
    
    Environment Variables:
        DATABASE_URL (required): PostgreSQL connection string
        OLLAMA_BASE_URL (optional): Ollama API base URL
        OLLAMA_EMBEDDING_MODEL (optional): Embedding model name
        DOCUMENTS_TABLE (optional): Table name (default: documents)
        CONTENT_COLUMN (optional): Content column name (default: content)
    
    Args:
        query: The search query string
        top_n: Number of top similar documents to return (default: 3)
        
    Returns:
        A list of dictionaries containing title and content of most similar documents
        
    Example:
        results = await search_documents("machine learning", top_n=5)
        for doc in results:
            print(f"{doc['title']}: {doc['content'][:100]}...")
    """
    # Check for required environment variables
    database_url = os.getenv('DATABASE_URL')
    table_name = os.getenv('DOCUMENTS_TABLE', 'documents')
    content_column = os.getenv('CONTENT_COLUMN', 'content')
    title_column = os.getenv('TITLE_COLUMN', 'title')
    
    if not database_url:
        # Setup environment if not set
        os.environ["DATABASE_URL"] = "postgresql://postgres:2336@localhost:5432/pagila"
        os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
        os.environ["OLLAMA_EMBEDDING_MODEL"] = "mxbai-embed-large"
        database_url = os.environ["DATABASE_URL"]
    
    # Try vector search first
    async def search_vector_documents():
        try:
            # Generate embedding for query
            print(f"🧠 Generating embedding for query: '{query[:50]}...'")
            query_embedding = await get_embedding(query)
            
            if not query_embedding:
                print("⚠️  No query embedding generated, falling back to text search")
                return None
                
            # Connect to database
            conn_params = psycopg2.extensions.parse_dsn(database_url)
            conn_params.update({
                'options': '-c default_transaction_read_only=on -c statement_timeout=10000'
            })
            
            conn = psycopg2.connect(**conn_params)
            
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # Check if we have embeddings in the table
                cursor.execute(f"SELECT COUNT(*) as embedding_count FROM {table_name} WHERE embedding IS NOT NULL;")
                embedding_count = cursor.fetchone()['embedding_count']
                
                if embedding_count == 0:
                    print("⚠️  No document embeddings found, falling back to text search")
                    conn.close()
                    return None
                
                print(f"🔍 Using vector search on {embedding_count} embedded documents")
                
                # Convert query embedding to pgvector format
                embedding_str = f"[{','.join(map(str, query_embedding))}]"
                print(f"🔢 Query embedding vector length: {len(query_embedding)}")
                print(f"🔢 Query embedding sample: {query_embedding[:5]}...")
                
                # Vector similarity search using cosine distance
                vector_sql = f"""
                    SELECT {content_column},
                           COALESCE({title_column}, '') as title,
                           1 - (embedding <=> %s::vector) as similarity
                    FROM {table_name} 
                    WHERE embedding IS NOT NULL
                    ORDER BY similarity DESC
                    LIMIT %s;
                """
                
                print(f"🔍 Executing vector search SQL...")
                print(f"📝 SQL: {vector_sql}")
                try:
                    cursor.execute(vector_sql, (embedding_str, top_n))
                    results = cursor.fetchall()
                    print(f"🔍 Vector search returned {len(results) if results else 0} rows")
                except Exception as sql_error:
                    print(f"❌ SQL execution error: {sql_error}")
                    raise sql_error
                
                conn.close()
                
                if results:
                    print(f"✅ Vector search found {len(results)} results")
                    return [{"title": row["title"], "content": row[content_column]} for row in results]
                else:
                    print("⚠️  Vector search returned no results, falling back to text search")
                    return None
                    
        except Exception as e:
            print(f"❌ Vector search error: {e}")
            print("⚠️  Falling back to text search")
            return None
    
    # Text search fallback
    def search_text_documents():
        try:
            print(f"📝 Using text search for query: '{query[:50]}...'")
            
            # Connect to database with read-only settings
            conn_params = psycopg2.extensions.parse_dsn(database_url)
            conn_params.update({
                'options': '-c default_transaction_read_only=on -c statement_timeout=10000'
            })
            
            conn = psycopg2.connect(**conn_params)
            
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # Use PostgreSQL full-text search
                search_sql = f"""
                    SELECT {content_column},
                           COALESCE({title_column}, '') as title,
                           ts_rank(to_tsvector('english', COALESCE({title_column}, '') || ' ' || {content_column}), 
                                  plainto_tsquery('english', %s)) as rank
                    FROM {table_name} 
                    WHERE to_tsvector('english', COALESCE({title_column}, '') || ' ' || {content_column}) @@ plainto_tsquery('english', %s)
                    ORDER BY rank DESC
                    LIMIT %s;
                """
                
                cursor.execute(search_sql, (query, query, top_n))
                results = cursor.fetchall()
                
                conn.close()
                
                if results:
                    print(f"✅ Text search found {len(results)} results")
                    return [{"title": row["title"], "content": row[content_column]} for row in results]
                else:
                    print(f"❌ No documents found matching '{query}'")
                    return []
                    
        except Exception as e:
            print(f"❌ Text search error: {e}")
            return []
    
    # Try vector search first, fall back to text search
    vector_results = await search_vector_documents()
    if vector_results is not None:
        return vector_results
    else:
        return search_text_documents()


async def summarize_rows(rows: list[dict], question: str) -> str:
    """
    Summarizes query results into a human-readable format.
    
    For small result sets (≤20 rows), returns a formatted Markdown table.
    For larger result sets (>20 rows), uses OpenAI GPT or Ollama to generate a concise summary.
    
    Environment Variables:
        OPENAI_API_KEY (optional): OpenAI API key (if using OpenAI)
        OPENAI_MODEL (optional): Model to use for summarization (default: gpt-3.5-turbo)
        USE_OLLAMA (optional): Set to 'true' to use Ollama instead of OpenAI
        OLLAMA_HOST (optional): Ollama server URL (default: http://localhost:11434)
        OLLAMA_MODEL (optional): Model to use for summarization (default: mistral:7b-instruct)
    
    Args:
        rows: List of dictionaries representing query results
        question: The original question that was asked
        
    Returns:
        A string containing either a Markdown table or an AI-generated summary
        
    Raises:
        Exception: If API call fails for large result sets
        
    Example:
        # Small result set
        rows = [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]
        result = await summarize_rows(rows, "Who are the users?")
        # Returns a Markdown table
        
        # Large result set with OpenAI
        export OPENAI_API_KEY='your-key'
        rows = [{"product": f"Item {i}", "sales": i*100} for i in range(50)]
        result = await summarize_rows(rows, "What are the top selling products?")
        # Returns an AI-generated summary
    """
    if not rows:
        return "No results found."
    
    # For small result sets, return a Markdown table
    if len(rows) <= 20:
        # Get all unique column names from all rows
        all_columns = set()
        for row in rows:
            all_columns.update(row.keys())
        columns = sorted(list(all_columns))
        
        if not columns:
            return "No data columns found."
        
        # Build Markdown table
        markdown_lines = []
        
        # Header row
        header = "| " + " | ".join(columns) + " |"
        markdown_lines.append(header)
        
        # Separator row
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        markdown_lines.append(separator)
        
        # Data rows
        for row in rows:
            values = []
            for col in columns:
                value = row.get(col, "")
                # Convert to string and escape pipes
                value_str = str(value).replace("|", "\\|") if value is not None else ""
                values.append(value_str)
            data_row = "| " + " | ".join(values) + " |"
            markdown_lines.append(data_row)
        
        return "\n".join(markdown_lines)
    
    # For large result sets, use AI to summarize
    use_ollama = os.getenv('USE_OLLAMA', 'false').lower() == 'true'
    
    # Prepare a sample of the data for summarization
    sample_size = min(5, len(rows))
    sample_rows = rows[:sample_size]
    
    # Build a prompt for summarization
    summary_prompt = f"""You are analyzing query results for the question: "{question}"

The query returned {len(rows)} rows. Here's a sample of the first {sample_size} rows:

{json.dumps(sample_rows, indent=2)}

Please provide a concise summary of these results that answers the original question. Include:
1. Total number of rows returned
2. Key insights or patterns from the data
3. A direct answer to the original question if possible

Keep the summary brief but informative."""

    messages = [{"role": "user", "content": summary_prompt}]
    
    try:
        if use_ollama:
            # Use Ollama
            ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
            ollama_model = os.getenv('OLLAMA_MODEL', 'mistral:7b-instruct')
            
            ollama_payload = {
                "model": ollama_model,
                "messages": messages,
                "options": {
                    "temperature": 0.1
                },
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{ollama_host}/api/chat",
                    json=ollama_payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        # Fallback to basic summary if Ollama fails
                        return f"Query returned {len(rows)} rows. First few results:\n{json.dumps(sample_rows, indent=2)}"
                    
                    response_data = await response.json()
                    
                    # Extract the summary from Ollama's response
                    message = response_data.get('message', {})
                    summary = message.get('content', '')
        else:
            # Use OpenAI
            openai_api_key = os.getenv('OPENAI_API_KEY')
            openai_model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
            
            if not openai_api_key:
                return f"Query returned {len(rows)} rows. First few results:\n{json.dumps(sample_rows, indent=2)}"
            
            client = AsyncOpenAI(api_key=openai_api_key)
            
            response = await client.chat.completions.create(
                model=openai_model,
                messages=messages,
                temperature=0.1,
                max_tokens=500
            )
            
            summary = response.choices[0].message.content
        
        if summary:
            return summary.strip()
        else:
            # Fallback if no content
            return f"Query returned {len(rows)} rows. First few results:\n{json.dumps(sample_rows, indent=2)}"
            
    except Exception as e:
        # Fallback to basic summary if anything fails
        return f"Query returned {len(rows)} rows. First few results:\n{json.dumps(sample_rows, indent=2)}"


async def generate_final_answer(question: str) -> str:
    """
    Generate a comprehensive final answer by combining database results and external documents.
    
    This function orchestrates the complete workflow:
    1. Queries the database using ask_db()
    2. Summarizes database results using summarize_rows()
    3. Searches for relevant external documents using search_documents()
    4. Combines all information and uses OpenAI GPT or Ollama to generate a final answer
    
    Environment Variables:
        DATABASE_URL (required): PostgreSQL connection string
        OPENAI_API_KEY (optional): OpenAI API key (if using OpenAI)
        OPENAI_MODEL (optional): Chat model for final answer (default: gpt-3.5-turbo)
        USE_OLLAMA (optional): Set to 'true' to use Ollama instead of OpenAI
        OLLAMA_HOST (optional): Ollama server URL (default: http://localhost:11434)
        OLLAMA_MODEL (optional): Ollama model for final answer (default: mistral:7b-instruct)
        OLLAMA_EMBEDDING_MODEL (optional): Embedding model for document search (default: mxbai-embed-large)
    
    Args:
        question: The user's question requiring both database and document analysis
        
    Returns:
        A comprehensive final answer combining database results and external documents
        
    Raises:
        ValueError: If DATABASE_URL is not configured
        Exception: If any component fails (with graceful degradation)
        
    Example:
        # Set up environment for OpenAI
        export DATABASE_URL='postgresql://postgres:2336@localhost:5432/pagila'
        export OPENAI_API_KEY='your-openai-api-key'
        export OPENAI_MODEL='gpt-4'
        
        # OR set up environment for Ollama
        export DATABASE_URL='postgresql://postgres:2336@localhost:5432/pagila'
        export USE_OLLAMA='true'
        export OLLAMA_MODEL='mistral:7b-instruct'
        
        # Use the function
        answer = await generate_final_answer("What are the most popular film genres and their characteristics?")
        print(answer)
    """
    use_ollama = os.getenv('USE_OLLAMA', 'false').lower() == 'true'
    
    if use_ollama:
        ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        ollama_model = os.getenv('OLLAMA_MODEL', 'mistral:7b-instruct')
    else:
        openai_api_key = os.getenv('OPENAI_API_KEY')
        openai_model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    
    # Initialize components for the final answer
    db_summary = ""
    documents = []
    errors = []
    
    print(f"🔍 Processing question: {question}")
    
    # Step 1: Get database results and summarize them
    try:
        print("📊 Querying database...")
        db_rows = await ask_db(question)
        print(f"   Found {len(db_rows)} database results")
        
        print("📝 Summarizing database results...")
        db_summary = await summarize_rows(db_rows, question)
        print(f"   Generated summary ({len(db_summary)} chars)")
        
    except Exception as e:
        error_msg = f"Database query failed: {e}"
        errors.append(error_msg)
        print(f"   ❌ {error_msg}")
        db_summary = "No database results available due to query error."
    
    # Step 2: Search for relevant external documents
    try:
        print("🔍 Searching external documents...")
        documents = await search_documents(question, top_n=3)
        print(f"   Found {len(documents)} relevant documents")
        
    except Exception as e:
        error_msg = f"Document search failed: {e}"
        errors.append(error_msg)
        print(f"   ❌ {error_msg}")
        documents = []
    
    # Step 3: Build comprehensive prompt for final answer
    print("🤖 Generating final answer...")
    
    # Prepare the combined context
    context_parts = []
    
    # Add database summary
    if db_summary and db_summary.strip() != "No database results available due to query error.":
        context_parts.append(f"## Database Results\n\n{db_summary}")
    
    # Add external documents
    if documents:
        doc_section = "## External Documents\n\n"
        for i, doc in enumerate(documents, 1):
            # Truncate very long documents
            doc_content = doc.get('content', str(doc))
            truncated_doc = doc_content[:500] + "..." if len(doc_content) > 500 else doc_content
            doc_section += f"### Document {i}\n{truncated_doc}\n\n"
        context_parts.append(doc_section)
    
    # Build the final prompt
    if context_parts:
        combined_context = "\n".join(context_parts)
        
        final_prompt = f"""You are an AI assistant helping to answer questions using both database results and external documents.

Question: {question}

Available Information:
{combined_context}

Instructions:
1. Analyze both the database results and external documents
2. Provide a comprehensive answer that combines insights from both sources
3. If database and documents provide different perspectives, acknowledge both
4. Be specific and cite information when possible
5. If some information is missing or incomplete, acknowledge the limitations

Please provide a detailed, well-structured answer to the question."""

    else:
        # Fallback when no data is available
        final_prompt = f"""Question: {question}

Unfortunately, I was unable to retrieve relevant information from either the database or external documents due to the following issues:
{chr(10).join(['- ' + error for error in errors])}

Please check your environment configuration and try again. You may need to:
1. Set DATABASE_URL for database access
2. Set OPENAI_API_KEY for OpenAI access (or USE_OLLAMA=true for Ollama)
3. Run setup_document_search.py for document search functionality
4. Ensure Ollama is running locally (if using Ollama)"""

    # Step 4: Call API to generate the final answer
    messages = [{"role": "user", "content": final_prompt}]
    
    try:
        if use_ollama:
            # Use Ollama
            ollama_payload = {
                "model": ollama_model,
                "messages": messages,
                "options": {
                    "temperature": 0.2
                },
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{ollama_host}/api/chat",
                    json=ollama_payload,
                    timeout=aiohttp.ClientTimeout(total=90)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ollama API returned {response.status}: {error_text}")
                    
                    response_data = await response.json()
                    
                    # Extract the final answer
                    message = response_data.get('message', {})
                    final_answer = message.get('content', '')
        else:
            # Use OpenAI
            if not openai_api_key:
                raise Exception("OPENAI_API_KEY is required when USE_OLLAMA is not set to 'true'")
            
            client = AsyncOpenAI(api_key=openai_api_key)
            
            response = await client.chat.completions.create(
                model=openai_model,
                messages=messages,
                temperature=0.2,
                max_tokens=2000
            )
            
            final_answer = response.choices[0].message.content
        
        if final_answer:
            print(f"   ✅ Generated final answer ({len(final_answer)} chars)")
            return final_answer.strip()
        else:
            raise Exception("API returned empty response")
            
    except Exception as e:
        # Fallback to a basic combined response if API fails
        fallback_answer = f"""# Answer to: {question}

## Summary
I encountered an issue generating the AI-powered final answer: {e}

## Available Information

"""
        
        if db_summary and db_summary.strip() != "No database results available due to query error.":
            fallback_answer += f"### Database Results\n{db_summary}\n\n"
        
        if documents:
            fallback_answer += "### External Documents\n"
            for i, doc in enumerate(documents, 1):
                doc_content = doc.get('content', str(doc))
                truncated_doc = doc_content[:300] + "..." if len(doc_content) > 300 else doc_content
                fallback_answer += f"**Document {i}:** {truncated_doc}\n\n"
        
        if not db_summary and not documents:
            fallback_answer += "No information could be retrieved from database or documents.\n"
        
        print(f"   ⚠️  Using fallback answer ({len(fallback_answer)} chars)")
        return fallback_answer


if __name__ == '__main__':
    # Example usage of extract_schema:
    # Replace with your actual database connection details
    # DATABASE_URL = "postgresql://user:password@host:port/dbname"
    #
    # try:
    #     connection = psycopg2.connect(DATABASE_URL)
    #     db_schema = extract_schema(connection)
    #     import json
    #     print("--- Database Schema ---")
    #     print(json.dumps(db_schema, indent=2))
    #     connection.close()
    # except psycopg2.OperationalError as e:
    #     print(f"Could not connect to the database: {e}")

    # Example usage of build_prompt:
    sample_schema = {
        "film": [
            {"column": "film_id", "type": "integer"},
            {"column": "title", "type": "varchar"},
            {"column": "release_year", "type": "integer"},
        ],
        "actor": [
            {"column": "actor_id", "type": "integer"},
            {"column": "first_name", "type": "varchar"},
            {"column": "last_name", "type": "varchar"},
        ],
    }
    sample_question = "What are the titles of films released in the year 2006?"

    messages, functions = build_prompt(sample_question, sample_schema)

    import json
    print("\n--- Generated Prompt ---")
    print(messages[0]['content'])
    print("\n--- Generated Functions ---")
    print(json.dumps(functions, indent=2))
