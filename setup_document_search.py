#!/usr/bin/env python3
"""
Setup script for creating the documents table and testing the search_documents function.
"""

import asyncio
import os
import psycopg2
from psycopg2.extras import RealDictCursor
import aiohttp
import json

def setup_environment():
    """Set up environment variables for consistent operation"""
    os.environ["DATABASE_URL"] = "postgresql://postgres:2336@localhost:5432/pagila"
    os.environ["CHAT_MODEL"] = "mistral:7b-instruct"
    os.environ["EMBED_MODEL"] = "mxbai-embed-large"
    os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
    os.environ["OLLAMA_HOST"] = "http://localhost:11434"
    os.environ["OLLAMA_MODEL"] = "mistral:7b-instruct"
    os.environ["OLLAMA_EMBEDDING_MODEL"] = "mxbai-embed-large"

async def get_embedding(text: str) -> list:
    """Generate embedding for text using Ollama."""
    
    try:
        async with aiohttp.ClientSession() as session:
            embed_url = f"{os.getenv('OLLAMA_BASE_URL')}/api/embeddings"
            payload = {
                "model": os.getenv('OLLAMA_EMBEDDING_MODEL'),
                "prompt": text
            }
            
            async with session.post(embed_url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get('embedding', [])
                else:
                    print(f"❌ Embedding API error: {response.status}")
                    return []
                    
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return []

async def setup_documents_table():
    """Create the documents table with pgvector support."""
    
    # Setup environment first
    setup_environment()
    
    print("=== Setting up Documents Table ===\n")
    
    if not os.getenv('DATABASE_URL'):
        print("❌ DATABASE_URL not set")
        print("   Example: export DATABASE_URL='postgresql://user:password@localhost:5432/dbname'")
        return False
    
    try:
        conn = psycopg2.connect(os.getenv('DATABASE_URL'))
        conn.autocommit = True
        
        with conn.cursor() as cursor:
            # Install pgvector extension
            print("🔧 Installing pgvector extension...")
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                print("✅ pgvector extension ready")
            except psycopg2.Error as e:
                print(f"❌ Failed to install pgvector: {e}")
                return False
            
            # Create documents table WITH vector column
            print("📊 Creating documents table...")
            table_sql = """
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    title VARCHAR(255),
                    content TEXT NOT NULL,
                    embedding vector(1024),  -- mxbai-embed-large uses 1024 dimensions
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """
            cursor.execute(table_sql)
            print("✅ Documents table created (with vector search)")
            
            # Create vector similarity index
            print("🗂️ Creating vector similarity index...")
            try:
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS documents_embedding_idx 
                    ON documents USING ivfflat (embedding vector_cosine_ops) 
                    WITH (lists = 100);
                """)
                print("✅ Vector similarity index created")
            except psycopg2.Error as e:
                print(f"⚠️  Vector index creation failed: {e}")
                print("   This is normal if the table is empty. Index will be created when you add documents.")
            
            # Also create text search index as fallback
            print("🗂️ Creating full-text search index as fallback...")
            try:
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS documents_content_search_idx 
                    ON documents USING gin(to_tsvector('english', COALESCE(title, '') || ' ' || content));
                """)
                print("✅ Full-text search index created as fallback")
            except psycopg2.Error as e:
                print(f"⚠️  Text search index creation failed: {e}")
            
            # Insert sample documents WITH embeddings
            print("\n🧠 Processing sample documents with embeddings...")
            sample_docs = [
                ("Machine Learning Basics", 
                 "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or classifications based on those patterns."),
                
                ("Python Programming", 
                 "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in web development, data science, artificial intelligence, and automation. Python's syntax allows developers to express concepts in fewer lines of code."),
                
                ("Database Systems", 
                 "A database is an organized collection of structured information stored electronically in a computer system. Database management systems (DBMS) are software applications that interact with users, applications, and the database itself to capture and analyze data."),
                
                ("Natural Language Processing", 
                 "Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human language. It involves developing algorithms and models that can understand, interpret, and generate human language in a valuable way."),
                
                ("Web Development", 
                 "Web development involves creating and maintaining websites and web applications. It includes front-end development (user interface), back-end development (server-side logic), and database management. Modern web development uses frameworks and tools to create responsive, interactive web experiences.")
            ]
            
            # Generate embeddings for each document
            for i, (title, content) in enumerate(sample_docs, 1):
                print(f"   Processing document {i}/{len(sample_docs)}: {title[:50]}...")
                
                try:
                    # Generate embedding
                    embedding = await get_embedding(content)
                    if embedding:
                        # Convert to pgvector format
                        embedding_str = f"[{','.join(map(str, embedding))}]"
                        
                        cursor.execute("""
                            INSERT INTO documents (title, content, embedding) 
                            VALUES (%s, %s, %s) 
                            ON CONFLICT DO NOTHING
                        """, (title, content, embedding_str))
                        print(f"   ✅ Document {i} embedded and stored")
                    else:
                        # Fallback without embedding
                        cursor.execute("""
                            INSERT INTO documents (title, content) 
                            VALUES (%s, %s) 
                            ON CONFLICT DO NOTHING
                        """, (title, content))
                        print(f"   ⚠️  Document {i} stored without embedding")
                        
                except Exception as e:
                    print(f"   ❌ Error processing document {i}: {e}")
                    # Store without embedding as fallback
                    cursor.execute("""
                        INSERT INTO documents (title, content) 
                        VALUES (%s, %s) 
                        ON CONFLICT DO NOTHING
                    """, (title, content))
            
            # Check how many documents we have
            cursor.execute("SELECT COUNT(*) FROM documents;")
            count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL;")
            embedded_count = cursor.fetchone()[0]
            print(f"✅ Documents table ready with {count} documents ({embedded_count} with embeddings)")
            
            
        conn.close()
        return True
        
    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
        return False


async def generate_embeddings_for_documents():
    """Generate embeddings for documents that don't have them yet."""
    
    print("\n=== Generating Embeddings for Documents ===\n")
    
    try:
        conn = psycopg2.connect(os.getenv('DATABASE_URL'))
        
        with conn.cursor() as cursor:
            # Find documents without embeddings
            cursor.execute("SELECT id, title, content FROM documents WHERE embedding IS NULL;")
            documents = cursor.fetchall()
            
            if not documents:
                print("✅ All documents already have embeddings!")
                return True
            
            print(f"📊 Found {len(documents)} documents without embeddings")
            
            for doc_id, title, content in documents:
                print(f"   Processing: {title[:50]}...")
                
                try:
                    embedding = await get_embedding(content)
                    if embedding:
                        embedding_str = f"[{','.join(map(str, embedding))}]"
                        cursor.execute(
                            "UPDATE documents SET embedding = %s WHERE id = %s;",
                            (embedding_str, doc_id)
                        )
                        print(f"   ✅ Embedding generated for document {doc_id}")
                    else:
                        print(f"   ⚠️  Failed to generate embedding for document {doc_id}")
                        
                except Exception as e:
                    print(f"   ❌ Error processing document {doc_id}: {e}")
            
            conn.commit()
            
            # Final count
            cursor.execute("SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL;")
            embedded_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM documents;")
            total_count = cursor.fetchone()[0]
            
            print(f"✅ Embedding generation complete: {embedded_count}/{total_count} documents have embeddings")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        return False


async def test_document_search():
    """Test the document search functionality using PostgreSQL full-text search."""
    
    print("\n=== Testing Text-Based Document Search ===\n")
    
    try:
        conn = psycopg2.connect(os.getenv('DATABASE_URL'))
        
        test_queries = [
            "programming",
            "artificial intelligence",
            "database",
            "language"
        ]
        
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            for query in test_queries:
                print(f"🔍 Query: '{query}'")
                
                # Use PostgreSQL full-text search
                search_sql = """
                    SELECT title, content,
                           ts_rank(to_tsvector('english', title || ' ' || content), 
                                  plainto_tsquery('english', %s)) as rank
                    FROM documents 
                    WHERE to_tsvector('english', title || ' ' || content) @@ plainto_tsquery('english', %s)
                    ORDER BY rank DESC
                    LIMIT 2;
                """
                
                cursor.execute(search_sql, (query, query))
                results = cursor.fetchall()
                
                print(f"   Found {len(results)} relevant documents:")
                for i, doc in enumerate(results, 1):
                    preview = doc['content'][:80] + "..." if len(doc['content']) > 80 else doc['content']
                    print(f"   {i}. {doc['title']}: {preview}")
                print()
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Document search test failed: {e}")


async def main():
    """Main setup and test function."""
    
    print("🚀 Document Search Setup and Test\n")
    
    # Step 1: Setup database
    if not await setup_documents_table():
        print("❌ Database setup failed")
        return
    
    # Step 2: Generate embeddings
    if not await generate_embeddings_for_documents():
        print("❌ Embedding generation failed")
        return
    
    # Step 3: Test search
    await test_document_search()
    
    print("\n✅ Setup complete! You can now use search_documents() function.")
    print("\nExample usage:")
    print("from schema_tools import search_documents")
    print("results = await search_documents('your search query here')")


if __name__ == "__main__":
    asyncio.run(main())
