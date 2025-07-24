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
    schema_text = "\n".join(schema_lines)

    prompt = f"""{schema_text}

{question}

You must use the generate_sql function to provide a PostgreSQL SELECT query that answers the question. 
The query parameter must contain only the SQL query string.
Example: {{"query": "SELECT title FROM film LIMIT 10"}}

Use only the given schema tables. Generate a valid PostgreSQL SELECT query."""

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
    Asks the database a question by generating SQL via local Ollama and executing it safely.
    
    This function connects to a local Ollama server via HTTP API (no Python ollama package needed).
    
    Environment Variables:
        DATABASE_URL (required): PostgreSQL connection string
        OLLAMA_HOST (optional): Ollama server URL (default: http://localhost:11434)
        OLLAMA_MODEL (optional): Model to use (default: mistral:7b-instruct)
    
    Args:
        question: The natural language question to ask the database.
        
    Returns:
        A list of dictionaries representing the query results.
        
    Raises:
        ValueError: If no database URL is configured.
        SQLValidationError: If the generated SQL is unsafe or invalid.
        psycopg2.Error: If database connection or execution fails.
        Exception: If Ollama API call fails or returns unexpected format.
        
    Example:
        # Set up environment
        export DATABASE_URL='postgresql://postgres:2336@localhost:5432/pagila'
        export OLLAMA_MODEL='mistral:7b-instruct'  # if you don't have mistral
        
        # Start Ollama (in another terminal)
        ollama serve
        
        # Use the function
        result = await ask_db("How many users are in the database?")
    """
    # Check for required environment variables
    database_url = os.getenv('DATABASE_URL')
    ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    ollama_model = os.getenv('OLLAMA_MODEL', 'mistral:7b-instruct')
    
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
    
    # Step 3: Call local Ollama via HTTP API
    ollama_payload = {
        "model": ollama_model,
        "messages": messages,
        "tools": functions,  # same schema as OpenAI functions
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
                timeout=aiohttp.ClientTimeout(total=300)  # 5 minutes for large models
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ollama API returned {response.status}: {error_text}")
                
                response_data = await response.json()
                
    except aiohttp.ClientError as e:
        raise Exception(f"Failed to connect to Ollama at {ollama_host}: {e}")
    except Exception as e:
        raise Exception(f"Ollama API call failed: {e}")
    
    # Step 4: Extract the query from tool_calls
    message = response_data.get('message', {})
    tool_calls = message.get('tool_calls', [])
    
    if not tool_calls:
        raise Exception("Ollama did not return any tool calls")
    
    if len(tool_calls) == 0:
        raise Exception("Ollama returned empty tool calls")
    
    first_tool_call = tool_calls[0]
    
    # Handle Ollama's tool call structure
    try:
        tool_function = first_tool_call.get('function', {})
        function_name = tool_function.get('name', '')
        
        # Accept any function that looks like it's for SQL generation
        if not function_name or function_name.lower() in ['', 'none']:
            raise Exception(f"Ollama returned invalid function name: {function_name}")
        
        # Parse the arguments (they might be a JSON string)
        arguments = tool_function.get('arguments', {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        
        # Try different possible key names for the query
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
    
    # Step 5: Pass through verify_sql
    try:
        safe_query = verify_sql(generated_query)
    except SQLValidationError:
        # Re-raise validation errors as-is
        raise
    
    if safe_query is None:
        raise UnsafeSQLError("Generated query was deemed unsafe by verify_sql")
    
    # Step 6: Execute SQL with READ-ONLY connection and statement_timeout
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


async def search_documents(query: str, top_n: int = 3) -> list[str]:
    """
    Search for similar documents using Ollama embeddings and PostgreSQL vector similarity.
    
    This function:
    1. Uses Ollama's /api/embeddings to convert the query into a vector
    2. Queries PostgreSQL database to find top N most similar documents
    3. Returns a list of the top N document contents
    
    Prerequisites:
    - PostgreSQL with pgvector extension installed
    - A table with vector embeddings (e.g., 'documents' table with 'content' and 'embedding' columns)
    - Ollama running locally with an embedding model
    
    Environment Variables:
        DATABASE_URL (required): PostgreSQL connection string
        OLLAMA_HOST (optional): Ollama server URL (default: http://localhost:11434)
        OLLAMA_EMBEDDING_MODEL (optional): Embedding model (default: mxbai-embed-large)
        DOCUMENTS_TABLE (optional): Table name (default: documents)
        CONTENT_COLUMN (optional): Content column name (default: content)
        EMBEDDING_COLUMN (optional): Embedding column name (default: embedding)
    
    Args:
        query: The search query string
        top_n: Number of top similar documents to return (default: 3)
        
    Returns:
        A list of strings containing the top N most similar document contents
        
    Raises:
        ValueError: If required environment variables are missing
        Exception: If Ollama API call fails or database query fails
        
    Example:
        # Set up environment
        export DATABASE_URL='postgresql://postgres:2336@localhost:5432/pagila'
        export OLLAMA_EMBEDDING_MODEL='mxbai-embed-large'
        
        # Use the function
        results = await search_documents("machine learning algorithms", top_n=5)
        for doc in results:
            print(doc[:100] + "...")
    """
    # Check for required environment variables
    database_url = os.getenv('DATABASE_URL')
    ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    embedding_model = os.getenv('OLLAMA_EMBEDDING_MODEL', 'mxbai-embed-large')
    table_name = os.getenv('DOCUMENTS_TABLE', 'documents')
    content_column = os.getenv('CONTENT_COLUMN', 'content')
    embedding_column = os.getenv('EMBEDDING_COLUMN', 'embedding')
    
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is required")
    
    # Step 1: Use Ollama's /api/embeddings to convert query into vector
    ollama_payload = {
        "model": embedding_model,
        "prompt": query
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{ollama_host}/api/embeddings",
                json=ollama_payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ollama embeddings API returned {response.status}: {error_text}")
                
                embedding_response = await response.json()
                query_embedding = embedding_response.get('embedding')
                
                if not query_embedding:
                    raise Exception("Ollama did not return an embedding")
                    
    except aiohttp.ClientError as e:
        raise Exception(f"Failed to connect to Ollama at {ollama_host}: {e}")
    except Exception as e:
        raise Exception(f"Ollama embeddings API call failed: {e}")
    
    # Step 2: Query PostgreSQL database for top N most similar documents
    def search_similar_documents():
        try:
            # Connect to database with read-only settings
            conn_params = psycopg2.extensions.parse_dsn(database_url)
            conn_params.update({
                'options': '-c default_transaction_read_only=on -c statement_timeout=10000'
            })
            
            conn = psycopg2.connect(**conn_params)
            
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # Convert Python list to PostgreSQL vector format
                    vector_str = '[' + ','.join(map(str, query_embedding)) + ']'
                    
                    # SQL query to find most similar documents using cosine similarity
                    # Note: This assumes pgvector extension is installed and the embedding column is of type vector
                    similarity_query = f"""
                        SELECT {content_column},
                               1 - ({embedding_column} <=> %s::vector) AS similarity
                        FROM {table_name}
                        WHERE {embedding_column} IS NOT NULL
                        ORDER BY {embedding_column} <=> %s::vector
                        LIMIT %s;
                    """
                    
                    cursor.execute(similarity_query, (vector_str, vector_str, top_n))
                    rows = cursor.fetchall()
                    
                    # Extract just the content from the results
                    return [row[content_column] for row in rows]
                    
            finally:
                conn.close()
                
        except psycopg2.Error as e:
            raise Exception(f"Database search failed: {e}")
        except Exception as e:
            raise Exception(f"Document search failed: {e}")
    
    # Step 3: Execute database search using asyncio.to_thread
    similar_documents = await asyncio.to_thread(search_similar_documents)
    
    return similar_documents


async def summarize_rows(rows: list[dict], question: str) -> str:
    """
    Summarizes query results into a human-readable format.
    
    For small result sets (≤20 rows), returns a formatted Markdown table.
    For larger result sets (>20 rows), uses Ollama to generate a concise summary.
    
    Environment Variables:
        OLLAMA_HOST (optional): Ollama server URL (default: http://localhost:11434)
        OLLAMA_MODEL (optional): Model to use for summarization (default: mistral:7b-instruct)
    
    Args:
        rows: List of dictionaries representing query results
        question: The original question that was asked
        
    Returns:
        A string containing either a Markdown table or an AI-generated summary
        
    Raises:
        Exception: If Ollama API call fails for large result sets
        
    Example:
        # Small result set
        rows = [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]
        result = await summarize_rows(rows, "Who are the users?")
        # Returns a Markdown table
        
        # Large result set
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
    
    # For large result sets, use Ollama to summarize
    ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    ollama_model = os.getenv('OLLAMA_MODEL', 'mistral:7b-instruct')
    
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
    
    ollama_payload = {
        "model": ollama_model,
        "messages": messages,
        "options": {
            "temperature": 0.1
        },
        "stream": False
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{ollama_host}/api/chat",
                json=ollama_payload,
                timeout=aiohttp.ClientTimeout(total=120)  # Increased timeout for mistral
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    # Fallback to basic summary if Ollama fails
                    return f"Query returned {len(rows)} rows. First few results:\n{json.dumps(sample_rows, indent=2)}"
                
                response_data = await response.json()
                
                # Extract the summary from Ollama's response
                message = response_data.get('message', {})
                summary = message.get('content', '')
                
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
    4. Combines all information and uses Ollama to generate a final answer
    
    Environment Variables:
        DATABASE_URL (required): PostgreSQL connection string
        OLLAMA_HOST (optional): Ollama server URL (default: http://localhost:11434)
        OLLAMA_MODEL (optional): Chat model for final answer (default: mistral:7b-instruct)
        OLLAMA_EMBEDDING_MODEL (optional): Embedding model for document search (default: mxbai-embed-large)
    
    Args:
        question: The user's question requiring both database and document analysis
        
    Returns:
        A comprehensive final answer combining database results and external documents
        
    Raises:
        ValueError: If DATABASE_URL is not configured
        Exception: If any component fails (with graceful degradation)
        
    Example:
        # Set up environment
        export DATABASE_URL='postgresql://postgres:2336@localhost:5432/pagila'
        export OLLAMA_MODEL='mistral:7b-instruct'
        
        # Use the function
        answer = await generate_final_answer("What are the most popular film genres and their characteristics?")
        print(answer)
    """
    ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    ollama_model = os.getenv('OLLAMA_MODEL', 'mistral:7b-instruct')
    
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
            truncated_doc = doc[:500] + "..." if len(doc) > 500 else doc
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
2. Run setup_document_search.py for document search functionality
3. Ensure Ollama is running locally"""

    # Step 4: Call Ollama to generate the final answer
    messages = [{"role": "user", "content": final_prompt}]
    
    ollama_payload = {
        "model": ollama_model,
        "messages": messages,
        "options": {
            "temperature": 0.2  # Slightly more creative for final answers
        },
        "stream": False
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{ollama_host}/api/chat",
                json=ollama_payload,
                timeout=aiohttp.ClientTimeout(total=90)  # Longer timeout for complex answers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ollama API returned {response.status}: {error_text}")
                
                response_data = await response.json()
                
                # Extract the final answer
                message = response_data.get('message', {})
                final_answer = message.get('content', '')
                
                if final_answer:
                    print(f"   ✅ Generated final answer ({len(final_answer)} chars)")
                    return final_answer.strip()
                else:
                    raise Exception("Ollama returned empty response")
                    
    except Exception as e:
        # Fallback to a basic combined response if Ollama fails
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
                truncated_doc = doc[:300] + "..." if len(doc) > 300 else doc
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
