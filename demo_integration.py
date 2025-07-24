#!/usr/bin/env python3
"""
Integration example showing how to use ask_db with summarize_rows.
This demonstrates the complete workflow from question to summarized answer.
"""

import asyncio
import os
from schema_tools import ask_db, summarize_rows, generate_final_answer


async def ask_and_summarize(question: str) -> str:
    """
    Ask the database a question and return a summarized response.
    
    Args:
        question: The natural language question to ask
        
    Returns:
        A formatted summary of the results
    """
    try:
        # Step 1: Get raw results from the database
        rows = await ask_db(question)
        
        # Step 2: Summarize the results
        summary = await summarize_rows(rows, question)
        
        return summary
        
    except Exception as e:
        return f"Error: {e}"


async def demo_complete_workflow():
    """Demonstrate the complete workflow using generate_final_answer."""
    
    print("=== Complete Workflow Demo (generate_final_answer) ===")
    print("This function combines database results + external documents + AI analysis")
    print()
    
    # Check if environment is set up
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("❌ DATABASE_URL not set. Please set it and try again.")
        print("Example: export DATABASE_URL='postgresql://user:password@localhost:5432/dbname'")
        return
    
    print("✅ DATABASE_URL is set")
    print(f"Database: {database_url.split('@')[-1] if '@' in database_url else 'unknown'}")
    print()
    
    # Test questions that benefit from combined database + document analysis
    complex_questions = [
        "What are the most popular film genres in our database and what defines these genres?",
        "How do customer rental patterns relate to film characteristics and industry trends?",
        "What can you tell me about action movies - both from our data and general information?"
    ]
    
    for i, question in enumerate(complex_questions, 1):
        print(f"--- Complex Question {i}: {question} ---")
        
        try:
            # Use the complete workflow
            final_answer = await generate_final_answer(question)
            print(f"Complete Answer:\n{final_answer}")
            
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n" + "="*80 + "\n")
        
        # Add a delay between questions
        await asyncio.sleep(2)


async def demo_integration():
    """Demonstrate the integration between ask_db and summarize_rows."""
    
    print("=== ask_db + summarize_rows Integration Demo ===")
    print("This demo requires:")
    print("1. DATABASE_URL environment variable set")
    print("2. PostgreSQL database with data")
    print("3. Ollama running locally")
    print()
    
    # Check if environment is set up
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("❌ DATABASE_URL not set. Please set it and try again.")
        print("Example: export DATABASE_URL='postgresql://user:password@localhost:5432/dbname'")
        return
    
    print("✅ DATABASE_URL is set")
    print(f"Database: {database_url.split('@')[-1] if '@' in database_url else 'unknown'}")
    print()
    
    # Sample questions to test
    questions = [
        "How many tables are in the database?",
        "What are the first 5 films in the database?",
        "How many actors are there?",
        "What are all the film categories?",
        "Show me films from the year 2006"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"--- Question {i}: {question} ---")
        
        try:
            summary = await ask_and_summarize(question)
            print(f"Answer:\n{summary}")
            
        except Exception as e:
            print(f"Error: {e}")
        
        print()
        
        # Add a small delay between questions
        await asyncio.sleep(1)


async def test_with_mock_data():
    """Test summarize_rows with mock data when no database is available."""
    print("=== Testing with Mock Data (no database required) ===")
    
    # Mock small result set
    small_result = [
        {"film_id": 1, "title": "Academy Dinosaur", "release_year": 2006},
        {"film_id": 2, "title": "Ace Goldfinger", "release_year": 2006},
        {"film_id": 3, "title": "Adaptation Holes", "release_year": 2006}
    ]
    
    print("Small result set (should be Markdown table):")
    summary = await summarize_rows(small_result, "What are some films from 2006?")
    print(summary)
    print()
    
    # Mock large result set
    large_result = []
    for i in range(25):
        large_result.append({
            "film_id": i + 1,
            "title": f"Film Title {i + 1}",
            "release_year": 2006,
            "rating": "PG-13" if i % 2 == 0 else "R",
            "length": 90 + (i * 2)
        })
    
    print("Large result set (should use AI summarization or fallback):")
    summary = await summarize_rows(large_result, "What films do we have in our database?")
    print(summary)


async def main():
    """Run the integration demo."""
    
    print("Schema Tools Integration Demo")
    print("=" * 50)
    print()
    
    # Test with mock data first (always works)
    await test_with_mock_data()
    
    print("\n" + "=" * 50)
    print()
    
    # Test with real database (requires setup)
    await demo_integration()
    
    print("\n" + "=" * 50)
    print()
    
    # Test complete workflow (requires setup)
    await demo_complete_workflow()
    
    print("Demo completed!")


if __name__ == "__main__":
    asyncio.run(main())
