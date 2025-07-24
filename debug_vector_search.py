#!/usr/bin/env python3
"""
Debug script for vector search SQL
"""

import psycopg2
import os
from psycopg2.extras import RealDictCursor

def debug_vector_search():
    """Debug vector search step by step"""
    
    # Environment setup
    os.environ["DATABASE_URL"] = "postgresql://postgres:2336@localhost:5432/pagila"
    database_url = os.environ["DATABASE_URL"]
    
    try:
        conn = psycopg2.connect(database_url)
        
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Step 1: Check if documents table exists
            print("1. Checking documents table...")
            cursor.execute("SELECT COUNT(*) as doc_count FROM documents;")
            doc_count = cursor.fetchone()['doc_count']
            print(f"   Documents count: {doc_count}")
            
            # Step 2: Check embeddings
            cursor.execute("SELECT COUNT(*) as embedding_count FROM documents WHERE embedding IS NOT NULL;")
            embedding_count = cursor.fetchone()['embedding_count']
            print(f"   Documents with embeddings: {embedding_count}")
            
            # Step 3: Check one embedding format
            cursor.execute("SELECT id, embedding FROM documents WHERE embedding IS NOT NULL LIMIT 1;")
            sample = cursor.fetchone()
            if sample:
                print(f"   Sample embedding ID: {sample['id']}")
                print(f"   Sample embedding type: {type(sample['embedding'])}")
                print(f"   Sample embedding preview: {str(sample['embedding'])[:100]}...")
                
                # Parse the embedding to check its actual dimension
                import json
                try:
                    if isinstance(sample['embedding'], str):
                        # Remove brackets and split
                        embedding_str = sample['embedding'].strip('[]')
                        embedding_values = [float(x.strip()) for x in embedding_str.split(',')]
                        print(f"   Actual embedding dimension: {len(embedding_values)}")
                        print(f"   Sample values: {embedding_values[:5]}")
                    else:
                        print(f"   Embedding is not a string: {sample['embedding']}")
                except Exception as parse_error:
                    print(f"   ❌ Could not parse embedding: {parse_error}")
            
            # Step 4: Test vector query manually
            print("\n2. Testing vector search SQL...")
            
            # Use a real embedding from the database
            cursor.execute("SELECT embedding FROM documents WHERE embedding IS NOT NULL LIMIT 1;")
            real_embedding = cursor.fetchone()['embedding']
            
            # Also test with the same embedding (should give perfect similarity)
            vector_sql = """
                SELECT content,
                       COALESCE(title, '') as title,
                       1 - (embedding <=> %s::vector) as similarity
                FROM documents 
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT 3;
            """
            
            print(f"   SQL: {vector_sql}")
            print(f"   Using real embedding from database...")
            
            try:
                cursor.execute(vector_sql, (real_embedding, real_embedding))
                results = cursor.fetchall()
                print(f"   ✅ SQL executed successfully!")
                print(f"   Results count: {len(results)}")
                
                for i, row in enumerate(results):
                    print(f"   Result {i+1}: {row['title'][:50]}... (similarity: {row['similarity']:.4f})")
                    
            except Exception as sql_error:
                print(f"   ❌ SQL error: {sql_error}")
                print(f"   Error type: {type(sql_error)}")
                
                # Try to diagnose the issue
                print("\n3. Diagnosing the issue...")
                
                # Check pgvector extension
                try:
                    cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
                    ext_result = cursor.fetchone()
                    if ext_result:
                        print(f"   ✅ pgvector extension found: {ext_result['extname']}")
                    else:
                        print(f"   ❌ pgvector extension not found")
                except Exception as e:
                    print(f"   ❌ Error checking extension: {e}")
                
                # Check vector column type
                try:
                    cursor.execute("""
                        SELECT column_name, data_type, udt_name 
                        FROM information_schema.columns 
                        WHERE table_name = 'documents' AND column_name = 'embedding';
                    """)
                    col_info = cursor.fetchone()
                    if col_info:
                        print(f"   Column info: {dict(col_info)}")
                    else:
                        print(f"   ❌ embedding column not found")
                except Exception as e:
                    print(f"   ❌ Error checking column: {e}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        print(f"❌ Error type: {type(e)}")
        print(f"❌ Error args: {e.args}")
        import traceback
        print("❌ Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    debug_vector_search()
