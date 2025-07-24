#!/usr/bin/env python3
"""
Setup script for creating the documents table and testing the search_documents function.
"""

import asyncio
import os
import psycopg2
from psycopg2.extras import RealDictCursor


async def setup_documents_table():
    """Create the documents table with pgvector support."""
    
    print("=== Setting up Documents Table ===\n")
    
    if not os.getenv('DATABASE_URL'):
        print("❌ DATABASE_URL not set")
        print("   Example: export DATABASE_URL='postgresql://user:password@localhost:5432/dbname'")
        return False
    
    try:
        conn = psycopg2.connect(os.getenv('DATABASE_URL'))
        conn.autocommit = True
        
        with conn.cursor() as cursor:
            # Install pgvector extension if not exists
            print("🔧 Installing pgvector extension...")
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                print("✅ pgvector extension ready")
            except psycopg2.Error as e:
                print(f"❌ Failed to install pgvector: {e}")
                print("   You may need superuser privileges or install pgvector manually")
                return False
            
            # Create documents table
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
            print("✅ Documents table created")
            
            # Create index for vector similarity search
            print("🗂️ Creating vector similarity index...")
            try:
                cursor.execute("CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")
                print("✅ Vector index created")
            except psycopg2.Error as e:
                print(f"⚠️  Vector index creation failed: {e}")
                print("   This is normal if the table is empty. Index will be created when you add documents.")
            
            # Insert sample documents
            print("📝 Inserting sample documents...")
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
            
            for title, content in sample_docs:
                cursor.execute(
                    "INSERT INTO documents (title, content) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
                    (title, content)
                )
            
            # Check how many documents we have
            cursor.execute("SELECT COUNT(*) FROM documents;")
            count = cursor.fetchone()[0]
            print(f"✅ Documents table ready with {count} sample documents")
            
        conn.close()
        return True
        
    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
        return False


async def generate_embeddings_for_documents():
    """Generate embeddings for all documents that don't have them yet."""
    
    print("\n=== Generating Embeddings ===\n")
    
    try:
        import aiohttp
        
        ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        embedding_model = os.getenv('OLLAMA_EMBEDDING_MODEL', 'mxbai-embed-large')
        
        print(f"🤖 Using model: {embedding_model}")
        
        # Get documents without embeddings
        conn = psycopg2.connect(os.getenv('DATABASE_URL'))
        
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT id, title, content FROM documents WHERE embedding IS NULL;")
            documents = cursor.fetchall()
            
        if not documents:
            print("✅ All documents already have embeddings")
            conn.close()
            return True
        
        print(f"📄 Generating embeddings for {len(documents)} documents...")
        
        async with aiohttp.ClientSession() as session:
            for doc in documents:
                print(f"   Processing: {doc['title'][:50]}...")
                
                # Generate embedding
                payload = {
                    "model": embedding_model,
                    "prompt": doc['content']
                }
                
                try:
                    async with session.post(
                        f"{ollama_host}/api/embeddings",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            embedding_data = await response.json()
                            embedding = embedding_data.get('embedding')
                            
                            if embedding:
                                # Update document with embedding
                                vector_str = '[' + ','.join(map(str, embedding)) + ']'
                                
                                with conn.cursor() as update_cursor:
                                    update_cursor.execute(
                                        "UPDATE documents SET embedding = %s::vector WHERE id = %s;",
                                        (vector_str, doc['id'])
                                    )
                                    conn.commit()
                                
                                print(f"   ✅ Generated embedding for: {doc['title']}")
                            else:
                                print(f"   ❌ No embedding returned for: {doc['title']}")
                        else:
                            error_text = await response.text()
                            print(f"   ❌ API error for {doc['title']}: {response.status} - {error_text}")
                            
                except Exception as e:
                    print(f"   ❌ Failed to generate embedding for {doc['title']}: {e}")
        
        conn.close()
        print("✅ Embedding generation complete!")
        return True
        
    except ImportError:
        print("❌ aiohttp not available. Install with: pip install aiohttp")
        return False
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        return False


async def test_document_search():
    """Test the document search functionality."""
    
    print("\n=== Testing Document Search ===\n")
    
    try:
        from schema_tools import search_documents
        
        test_queries = [
            "programming languages and software development",
            "artificial intelligence and machine learning",
            "storing and organizing data",
            "understanding human language with computers"
        ]
        
        for query in test_queries:
            print(f"🔍 Query: '{query}'")
            
            results = await search_documents(query, top_n=2)
            
            print(f"   Found {len(results)} similar documents:")
            for i, doc in enumerate(results, 1):
                preview = doc[:80] + "..." if len(doc) > 80 else doc
                print(f"   {i}. {preview}")
            print()
            
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
