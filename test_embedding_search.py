#!/usr/bin/env python3
"""
Test script to debug embedding generation and vector search
"""

import asyncio
import psycopg2
import os
import aiohttp
from psycopg2.extras import RealDictCursor

async def get_embedding(text: str) -> list:
    """Generate embedding for text using Ollama."""
    try:
        base_url = "http://localhost:11434"
        model = "mxbai-embed-large"
        
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
                    return embedding
                else:
                    print(f"❌ Embedding API error: {response.status}")
                    return []
                    
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return []

async def test_embedding_and_search():
    """Test embedding generation and vector search"""
    
    os.environ["DATABASE_URL"] = "postgresql://postgres:2336@localhost:5432/pagila"
    database_url = os.environ["DATABASE_URL"]
    
    # Step 1: Generate embedding for test query
    print("1. Generating embedding for 'machine learning'...")
    query_embedding = await get_embedding("machine learning")
    print(f"   Generated embedding with {len(query_embedding)} dimensions")
    print(f"   Sample values: {query_embedding[:5]}")
    
    if not query_embedding:
        print("❌ No embedding generated, exiting")
        return
    
    # Step 2: Test with database
    try:
        conn = psycopg2.connect(database_url)
        
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            print("\n2. Testing vector search with new embedding...")
            
            # Convert to pgvector format
            embedding_str = f"[{','.join(map(str, query_embedding))}]"
            print(f"   Embedding string length: {len(embedding_str)}")
            print(f"   Embedding string preview: {embedding_str[:100]}...")
            
            # Test vector search
            vector_sql = """
                SELECT content,
                       COALESCE(title, '') as title,
                       1 - (embedding <=> %s::vector) as similarity
                FROM documents 
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT 3;
            """
            
            print(f"   Executing: {vector_sql}")
            cursor.execute(vector_sql, (embedding_str, embedding_str))
            results = cursor.fetchall()
            
            print(f"   ✅ Vector search executed!")
            print(f"   Results count: {len(results)}")
            
            if results:
                for i, row in enumerate(results):
                    print(f"   Result {i+1}: {row['title'][:50]}... (similarity: {row['similarity']:.4f})")
            else:
                print("   ❌ No results found")
                
                # Try without LIMIT
                print("   Trying without LIMIT...")
                simple_sql = """
                    SELECT title, 1 - (embedding <=> %s::vector) as similarity
                    FROM documents 
                    WHERE embedding IS NOT NULL;
                """
                
                cursor.execute(simple_sql, (embedding_str,))
                all_results = cursor.fetchall()
                print(f"   All results count: {len(all_results)}")
                
                if all_results:
                    # Sort by similarity manually
                    sorted_results = sorted(all_results, key=lambda x: x['similarity'], reverse=True)
                    print("   Top 3 results:")
                    for i, row in enumerate(sorted_results[:3]):
                        print(f"   Result {i+1}: {row['title'][:50]}... (similarity: {row['similarity']:.4f})")
                else:
                    print("   Still no results!")
                
                # Let's check what's in the database
                print("\n3. Checking database embeddings...")
                cursor.execute("SELECT title, embedding FROM documents WHERE embedding IS NOT NULL LIMIT 1;")
                sample = cursor.fetchone()
                
                if sample:
                    print(f"   Sample title: {sample['title']}")
                    
                    # Try direct comparison with database embedding
                    db_embedding = sample['embedding']
                    print(f"   DB embedding type: {type(db_embedding)}")
                    print(f"   DB embedding preview: {str(db_embedding)[:100]}...")
                    
                    # Test similarity with the exact same embedding
                    cursor.execute("""
                        SELECT 1 - (embedding <=> %s::vector) as similarity
                        FROM documents 
                        WHERE title = %s AND embedding IS NOT NULL;
                    """, (db_embedding, sample['title']))
                    
                    self_similarity = cursor.fetchone()
                    if self_similarity:
                        print(f"   Self-similarity (should be 1.0): {self_similarity['similarity']}")
                    
                    # Test with our new embedding vs this document
                    cursor.execute("""
                        SELECT 1 - (embedding <=> %s::vector) as similarity
                        FROM documents 
                        WHERE title = %s AND embedding IS NOT NULL;
                    """, (embedding_str, sample['title']))
                    
                    cross_similarity = cursor.fetchone()
                    if cross_similarity:
                        print(f"   Cross-similarity (query vs DB): {cross_similarity['similarity']}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_embedding_and_search())
