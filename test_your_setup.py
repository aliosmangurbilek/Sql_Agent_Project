#!/usr/bin/env python3
"""
Complete test for all schema_tools functions with your local Ollama setup.
"""

import asyncio
import os

def setup_environment():
    """Set up environment variables for consistent operation"""
    os.environ["DATABASE_URL"] = "postgresql://postgres:2336@localhost:5432/pagila"
    os.environ["CHAT_MODEL"] = "mistral:7b-instruct"
    os.environ["EMBED_MODEL"] = "mxbai-embed-large"
    os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
    os.environ["OLLAMA_HOST"] = "http://localhost:11434"
    os.environ["OLLAMA_MODEL"] = "mistral:7b-instruct"
    os.environ["OLLAMA_EMBEDDING_MODEL"] = "mxbai-embed-large"

async def main():
    """Main test function."""
    
    # Setup environment first
    setup_environment()
    
    print("=== Complete Ollama Integration Test ===\n")
    
    # Check current setup
    print("🔍 Current Setup:")
    print(f"   OLLAMA_HOST: {os.getenv('OLLAMA_HOST', 'http://localhost:11434 (default)')}")
    print(f"   OLLAMA_MODEL: {os.getenv('OLLAMA_MODEL', 'mistral:7b-instruct (default)')}")
    print(f"   OLLAMA_EMBEDDING_MODEL: {os.getenv('OLLAMA_EMBEDDING_MODEL', 'mxbai-embed-large (default)')}")
    print(f"   DATABASE_URL: {'✅ Set' if os.getenv('DATABASE_URL') else '❌ Not set'}")
    print()
    
    # Test available functions
    print("🛠️ Available Functions:")
    try:
        from schema_tools import ask_db, search_documents, verify_sql, summarize_rows, generate_final_answer
        print("   ✅ ask_db - Natural language to SQL queries")
        print("   ✅ search_documents - Vector similarity search")
        print("   ✅ verify_sql - SQL safety validation")
        print("   ✅ summarize_rows - Format results as tables or AI summaries")
        print("   ✅ generate_final_answer - Complete workflow combining all components")
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return
    
    print()
    
    # Suggest using an available model
    print("💡 Available models on your system:")
    print("   Chat models:")
    print("   - mistral:7b-instruct ⭐ (recommended for SQL generation)")
    print("   - gemma2:9b")
    print("   - gemma2:2b")
    print("   - qwen2.5:7b") 
    print("   - tinyllama:chat (lightweight option)")
    print()
    print("   Embedding models (for search_documents):")
    print("   - mxbai-embed-large (recommended, need to pull)")
    print("   - nomic-embed-text (alternative)")
    print()
    print("You can set specific models:")
    print("   export OLLAMA_MODEL='mistral:7b-instruct'")
    print("   export OLLAMA_EMBEDDING_MODEL='mxbai-embed-large'")
    print()
    print("For Pagila database:")
    print("   export DATABASE_URL='postgresql://postgres:2336@localhost:5432/pagila'")
    print()
    
    # Test functions if database is available
    if os.getenv('DATABASE_URL'):
        print("1. 🤖 Testing ask_db (natural language queries):")
        try:
            result = await ask_db("How many tables are in the database?")
            print(f"   ✅ ask_db returned {len(result)} rows")
            for i, row in enumerate(result[:3]):  # Show first 3 rows
                print(f"   Row {i+1}: {row}")
                
        except Exception as e:
            print(f"   ❌ ask_db Error: {e}")
            if "Ollama" in str(e):
                print("   💡 Try: export OLLAMA_MODEL='mistral:7b-instruct'")
        
        # Test search_documents
        print("\n2. 🔍 Testing search_documents (vector search):")
        try:
            docs = await search_documents("film categories database", top_n=2)
            print(f"   ✅ search_documents returned {len(docs)} documents")
            for i, doc in enumerate(docs):
                preview = doc[:80] + "..." if len(doc) > 80 else doc
                print(f"   {i}. {preview}")
                
        except Exception as e:
            print(f"   ❌ search_documents Error: {e}")
            if "pgvector" in str(e) or "documents" in str(e):
                print("   💡 Setup document search: python setup_document_search.py")
        
        # Test verify_sql
        print("\n3. 🛡️ Testing verify_sql (SQL validation):")
        try:
            test_queries = [
                "SELECT * FROM users",
                "DROP TABLE users",  # This should be rejected
            ]
            
            for test_query in test_queries:
                try:
                    safe_query = verify_sql(test_query)
                    print(f"   ✅ Safe: {test_query} → {safe_query}")
                except Exception as e:
                    print(f"   🚫 Blocked: {test_query} → {e}")
        
        except Exception as e:
            print(f"   ❌ verify_sql Error: {e}")
        
        # Test summarize_rows
        print("\n4. 📊 Testing summarize_rows (result formatting):")
        try:
            # Test with small result set
            small_rows = [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25}
            ]
            summary = await summarize_rows(small_rows, "Who are the users?")
            print(f"   ✅ Small result (table format): {len(summary)} chars")
            
            # Test with empty result
            empty_summary = await summarize_rows([], "No data query")
            print(f"   ✅ Empty result: {empty_summary}")
            
        except Exception as e:
            print(f"   ❌ summarize_rows Error: {e}")
        
        # Test generate_final_answer
        print("\n5. 🎯 Testing generate_final_answer (complete workflow):")
        try:
            test_question = "What information do we have about movies?"
            print(f"   Testing with: '{test_question}'")
            print("   Note: This will show progress as it runs each component...")
            
            final_answer = await generate_final_answer(test_question)
            
            # Show a preview of the answer
            preview = final_answer[:200] + "..." if len(final_answer) > 200 else final_answer
            print(f"   ✅ Generated final answer: {preview}")
            
        except Exception as e:
            print(f"   ❌ generate_final_answer Error: {e}")
            
    else:
        print("❌ DATABASE_URL not set. Set it to test with your database.")
        print("   Example: export DATABASE_URL='postgresql://postgres:2336@localhost:5432/pagila'")
    
    print("\n" + "="*60)
    print("📖 Complete Function Reference:")
    print("="*60)
    print()
    print("🤖 ask_db(question: str) -> list[dict]")
    print("   - Converts natural language to SQL using Ollama")
    print("   - Executes safely on read-only database")
    print("   - Environment: DATABASE_URL, OLLAMA_MODEL")
    print()
    print("🔍 search_documents(query: str, top_n: int = 3) -> list[str]")
    print("   - Vector similarity search using Ollama embeddings")
    print("   - Requires PostgreSQL with pgvector extension")
    print("   - Environment: DATABASE_URL, OLLAMA_EMBEDDING_MODEL")
    print()
    print("🛡️ verify_sql(query: str) -> str | None")
    print("   - Validates SQL safety (blocks INSERT/UPDATE/DELETE)")
    print("   - Adds LIMIT automatically")
    print("   - No environment variables needed")
    print()
    print("📊 summarize_rows(rows: list[dict], question: str) -> str")
    print("   - Formats ≤20 rows as Markdown table")
    print("   - Uses AI to summarize >20 rows")
    print("   - Environment: OLLAMA_HOST, OLLAMA_MODEL (for large result sets)")
    print()
    print("🎯 generate_final_answer(question: str) -> str")
    print("   - Complete workflow: database + documents + AI analysis")
    print("   - Combines ask_db, summarize_rows, search_documents")
    print("   - Environment: All above variables")
    print()
    print("🚀 Setup Commands:")
    print("   export DATABASE_URL='postgresql://postgres:2336@localhost:5432/pagila'")
    print("   export OLLAMA_MODEL='mistral:7b-instruct'")
    print("   python setup_document_search.py  # For search_documents")
    print("   python test_your_setup.py        # This test")
    print("   python example_complete_workflow.py  # Simple example")


if __name__ == "__main__":
    asyncio.run(main())
