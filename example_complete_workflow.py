#!/usr/bin/env python3
"""
Simple example of using generate_final_answer - the complete workflow function.
This demonstrates how to get comprehensive answers combining database and document sources.
"""

import asyncio
import os
from schema_tools import generate_final_answer


async def main():
    """Simple example of the complete workflow."""
    
    print("🎯 Generate Final Answer - Simple Example")
    print("=" * 50)
    print()
    
    # Check if environment is configured
    database_url = os.getenv('DATABASE_URL')
    print(f"Database configured: {'✅ Yes' if database_url else '❌ No'}")
    print()
    
    # Example question
    question = "What are the most popular film categories in the Pagila database?"
    
    print(f"Question: {question}")
    print("\nProcessing... (this combines database queries + document search + AI analysis)")
    print("-" * 60)
    
    try:
        # This single function call orchestrates everything:
        # 1. ask_db() to query the database
        # 2. summarize_rows() to format results
        # 3. search_documents() to find relevant documents  
        # 4. Ollama chat API to generate comprehensive answer
        answer = await generate_final_answer(question)
        
        print("\n🎉 Final Answer:")
        print("=" * 50)
        print(answer)
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure to:")
        print("   1. Set DATABASE_URL='postgresql://postgres:2336@localhost:5432/pagila'")
        print("   2. Run 'python setup_document_search.py' for document search")
        print("   3. Start Ollama server with 'ollama serve'")
        print("   4. Use available model: export OLLAMA_MODEL='mistral:7b-instruct'")
    
    print("\n✨ This demonstrates the complete workflow that combines:")
    print("   📊 Database results (structured data)")
    print("   📝 Smart summarization (tables or AI summaries)")
    print("   🔍 External documents (contextual information)")
    print("   🤖 AI analysis (comprehensive final answer)")


if __name__ == "__main__":
    asyncio.run(main())
