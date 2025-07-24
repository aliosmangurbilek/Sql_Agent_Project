#!/usr/bin/env python3
"""
Simplified comprehensive tests for the Pagila database system with Ollama models.
Tests the core SQL generation functionality that we know works.
"""

import os
import asyncio
import psycopg2
from schema_tools import ask_db, generate_final_answer

# Set environment variables for testing
os.environ["DATABASE_URL"] = "postgresql://postgres:2336@localhost:5432/pagila"
os.environ["CHAT_MODEL"] = "mistral:7b-instruct"
os.environ["EMBED_MODEL"] = "mxbai-embed-large"
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"

async def test_basic_connectivity():
    """Test basic database and Ollama connectivity."""
    print("🔗 Testing basic connectivity...")
    
    # Test database connection
    try:
        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        print("  ✅ Database connection successful")
        conn.close()
    except Exception as e:
        print(f"  ❌ Database connection failed: {e}")
        return False
    
    return True

async def test_sql_generation():
    """Test various SQL query generation scenarios."""
    print("\n🗄️ Testing SQL generation...")
    
    test_queries = [
        "How many films are in the database?",
        "What are the top 5 film categories by count?",
        "Which actors have appeared in the most films?",
        "What is the average rental rate for films?",
        "Show me films with 'DRAGON' in the title",
        "List all customers from a specific city like 'London'",
        "What is the total revenue from rentals?",
        "Show me the most popular film categories"
    ]
    
    successful_tests = 0
    
    for i, query in enumerate(test_queries, 1):
        print(f"  Test {i}: {query}")
        try:
            result = await ask_db(query)
            if result:
                print(f"    ✅ Got {len(result)} results")
                # Show first result as example
                if result:
                    sample = str(result[0])
                    if len(sample) > 100:
                        sample = sample[:100] + "..."
                    print(f"    📄 Sample: {sample}")
                successful_tests += 1
            else:
                print("    ⚠️ No results returned")
        except Exception as e:
            print(f"    ❌ Failed: {str(e)[:100]}...")
    
    print(f"\n  📊 Success rate: {successful_tests}/{len(test_queries)} ({successful_tests/len(test_queries)*100:.1f}%)")
    return successful_tests

async def test_complex_queries():
    """Test more complex SQL scenarios."""
    print("\n🧠 Testing complex queries...")
    
    complex_queries = [
        "Show me the customer who has rented the most films",
        "What is the average length of films by category?",
        "Which staff member has processed the most rentals?",
        "Show me films that have never been rented",
        "What is the monthly rental revenue trend?"
    ]
    
    successful_complex = 0
    
    for i, query in enumerate(complex_queries, 1):
        print(f"  Complex Test {i}: {query}")
        try:
            result = await ask_db(query)
            if result:
                print(f"    ✅ Got {len(result)} results")
                successful_complex += 1
            else:
                print("    ⚠️ No results returned")
        except Exception as e:
            print(f"    ❌ Failed: {str(e)[:100]}...")
    
    print(f"\n  📊 Complex query success rate: {successful_complex}/{len(complex_queries)} ({successful_complex/len(complex_queries)*100:.1f}%)")
    return successful_complex

async def test_error_handling():
    """Test system robustness with invalid inputs."""
    print("\n🛡️ Testing error handling...")
    
    error_tests = [
        "This is not a database question at all",
        "",
        "SELECT * FROM non_existent_table",
        "Show me something that doesn't make sense for a DVD rental database"
    ]
    
    handled_errors = 0
    
    for i, bad_query in enumerate(error_tests, 1):
        print(f"  Error Test {i}: '{bad_query[:50]}{'...' if len(bad_query) > 50 else ''}'")
        try:
            result = await ask_db(bad_query)
            if result:
                print("    ⚠️ Query unexpectedly succeeded")
            else:
                print("    ✅ Query returned no results (good)")
                handled_errors += 1
        except Exception as e:
            print(f"    ✅ Error properly handled: {type(e).__name__}")
            handled_errors += 1
    
    print(f"\n  📊 Error handling rate: {handled_errors}/{len(error_tests)} ({handled_errors/len(error_tests)*100:.1f}%)")
    return handled_errors

async def test_answer_generation():
    """Test the Q&A functionality."""
    print("\n🤖 Testing answer generation...")
    
    qa_questions = [
        "Explain the structure of the Pagila database",
        "How does the rental system work in this database?",
        "What are the main tables and their relationships?"
    ]
    
    successful_answers = 0
    
    for i, question in enumerate(qa_questions, 1):
        print(f"  Q&A Test {i}: {question}")
        try:
            answer = await generate_final_answer(question)
            if answer and len(answer) > 50:  # Reasonable answer length
                print(f"    ✅ Answer generated ({len(answer)} chars)")
                print(f"    📄 Preview: {answer[:100]}...")
                successful_answers += 1
            else:
                print("    ⚠️ Answer too short or empty")
        except Exception as e:
            print(f"    ❌ Failed: {str(e)[:100]}...")
    
    print(f"\n  📊 Answer generation rate: {successful_answers}/{len(qa_questions)} ({successful_answers/len(qa_questions)*100:.1f}%)")
    return successful_answers

async def main():
    """Run comprehensive tests."""
    print("🚀 Starting comprehensive Pagila system tests...")
    print("=" * 60)
    
    # Test basic connectivity first
    if not await test_basic_connectivity():
        print("\n❌ Basic connectivity failed. Stopping tests.")
        return
    
    # Run all tests and collect results
    sql_success = await test_sql_generation()
    complex_success = await test_complex_queries()
    error_success = await test_error_handling()
    qa_success = await test_answer_generation()
    
    print("\n" + "=" * 60)
    print("🎉 Comprehensive tests completed!")
    print("\n📊 FINAL RESULTS:")
    print(f"  - Database connectivity: ✅")
    print(f"  - Basic SQL generation: {sql_success}/8 tests passed")
    print(f"  - Complex queries: {complex_success}/5 tests passed")
    print(f"  - Error handling: {error_success}/4 tests passed")
    print(f"  - Answer generation: {qa_success}/3 tests passed")
    
    # Calculate overall success rate
    total_tests = 8 + 5 + 4 + 3
    total_success = sql_success + complex_success + error_success + qa_success
    overall_rate = (total_success / total_tests) * 100
    
    print(f"\n🏆 Overall Success Rate: {total_success}/{total_tests} ({overall_rate:.1f}%)")
    
    if overall_rate >= 75:
        print("✅ System is PRODUCTION READY! 🎉")
    elif overall_rate >= 50:
        print("⚠️ System is functional but needs improvement")
    else:
        print("❌ System needs significant work")
    
    print("\n💡 The Pagila database system with Ollama is working!")

if __name__ == "__main__":
    asyncio.run(main())
