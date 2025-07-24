#!/bin/bash
# Quick start script for Pagila Database AI Assistant

echo "🎯 Pagila Database AI Assistant - Quick Start"
echo "=============================================="
echo ""

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "❌ Ollama is not running. Please start Ollama first:"
    echo "   ollama serve"
    exit 1
fi

echo "✅ Docker and Ollama are running"
echo ""

# Start database
echo "🗄️ Starting PostgreSQL database..."
cd pagila/
docker-compose up -d

echo "⏳ Waiting for database to be ready..."
sleep 30

# Test database connection
echo "🧪 Testing database connection..."
if docker exec pagila-postgres psql -U postgres -d pagila -c "SELECT COUNT(*) FROM film;" >/dev/null 2>&1; then
    echo "✅ Database is ready"
else
    echo "❌ Database connection failed"
    exit 1
fi

cd ..

echo ""
echo "🚀 Starting AI Assistant interfaces..."
echo ""

# Start Streamlit Pro in background
echo "🎨 Starting Streamlit Pro (http://localhost:8502)..."
streamlit run app_pro.py --port 8502 > /dev/null 2>&1 &
STREAMLIT_PID=$!

# Start Flask API in background
echo "🔌 Starting Flask API (http://localhost:5000)..."
python flask_api.py > /dev/null 2>&1 &
FLASK_PID=$!

echo ""
echo "✅ All services started successfully!"
echo ""
echo "📱 Access your AI Assistant:"
echo "   🎨 Streamlit Pro:  http://localhost:8502  (Recommended)"
echo "   🔌 Flask API:      http://localhost:5000"
echo "   🗄️ pgAdmin:        http://localhost:5050"
echo ""
echo "💡 To stop all services:"
echo "   kill $STREAMLIT_PID $FLASK_PID"
echo "   cd pagila && docker-compose down"
echo ""
echo "🎉 Ready to explore your data with AI!"
