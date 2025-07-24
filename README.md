# 🎯 Pagila Database AI Assistant

**Production-ready Natural Language to SQL & Document Search Workflow using Local Ollama Models**

A comprehensive, modern system that combines AI-powered natural language processing with database querying using the Pagila sample database and local Ollama models.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13+-green.svg)](https://www.postgresql.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Latest-orange.svg)](https://ollama.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)](https://streamlit.io/)
[![Flask](https://img.shields.io/badge/Flask-Latest-blue.svg)](https://flask.palletsprojects.com/)

---

## 🚀 Quick Start

### 1. Start Database
```bash
cd pagila/
docker-compose up -d
```

### 2. Configure Python Environment
```bash
pip install streamlit flask flask-cors plotly pandas psycopg2-binary aiohttp
```

### 3. Launch Interfaces

#### 🚀 Quick Start (Recommended)
```bash
# One-click launcher with dependency checks
python run_app.py
```

#### 🔧 Manual Launch
```bash
# Streamlit Pro App (Main Interface)
streamlit run app_pro.py --server.port 8502

# Flask API + Web Interface
python flask_api.py
```

#### 📦 All Services at Once
```bash
# Start everything with one command
./start_all.sh
```

### 4. Access Your AI Assistant
- **🎨 Streamlit Pro**: http://localhost:8502 *(Main interactive interface)*
- **🔌 Flask API**: http://localhost:5000 *(For REST API and web interface)*

---

## 🌟 Key Features

### 🧠 AI-Powered Capabilities
- **Natural Language to SQL**: Ask questions in plain English
- **Intelligent Query Generation**: Mistral 7B Instruct model
- **Document Search**: MXBai embedding model for semantic search
- **Smart Error Recovery**: Automatic query refinement and validation

### 🎨 Multiple Interface Options
- **Streamlit Pro Dashboard**: Advanced analytics with real-time charts
- **Flask REST API**: RESTful endpoints for integration
- **Web Interface**: User-friendly browser interface
- **CLI Access**: Direct programmatic access

### 📊 Advanced Analytics
- **Real-time Visualization**: Interactive Plotly charts
- **Performance Metrics**: Query timing and success rates
- **Query History**: Track and replay previous queries
- **Data Export**: Download results in multiple formats

---

## 🔧 Usage Examples

### 🌐 Web Interface (Streamlit Pro)
- **URL**: http://localhost:8502
- **Features**: 
  - Multi-query mode (Fast SQL / AI Analysis / Hybrid)
  - Real-time visualization
  - Performance metrics
  - Query history
  - Dashboard and analysis tools

### 🔌 REST API (Flask)
- **Base URL**: http://localhost:5000
- **Web Interface**: http://localhost:5000
- **Endpoints**:
  ```bash
  # System status
  curl http://localhost:5000/api/status
  
  # Fast SQL query
  curl -X POST http://localhost:5000/api/query \
    -H "Content-Type: application/json" \
    -d '{"question": "How many films are in the database?"}'
  
  # AI detailed analysis
  curl -X POST http://localhost:5000/api/analyze \
    -H "Content-Type: application/json" \
    -d '{"question": "Analyze film categories"}'
  ```

### 💻 Programmatic Usage
```python
import asyncio
from schema_tools import ask_db, generate_final_answer

async def main():
    # Fast SQL query
    result = await ask_db("What is the longest film?")
    print("SQL Result:", result)
    
    # AI-powered analysis
    analysis = await generate_final_answer("Explain the Pagila database structure")
    print("AI Analysis:", analysis)

asyncio.run(main())
```

## 📊 Feature Comparison

| Feature | Streamlit Pro | Flask API | CLI |
|---------|---------------|-----------|-----|
| 🖥️ Web Interface | ✅ | ✅ | ❌ |
| 🔌 REST API | ❌ | ✅ | ❌ |
| 📊 Visualization | ✅✅ | ❌ | ❌ |
| 📈 Dashboard | ✅ | ❌ | ❌ |
| 📋 Query History | ✅ | ❌ | ❌ |
| ⚡ Performance Metrics | ✅ | ❌ | ❌ |
| 🔧 System Tools | ✅ | ❌ | ❌ |
| 💾 Data Download | ✅ | ❌ | ❌ |
| 🤖 AI Analysis | ✅ | ✅ | ✅ |
| 🚀 Fast SQL | ✅ | ✅ | ✅ |

## 🎯 Demo Queries

### 📊 Simple Queries
```
- "How many films are in the database?"
- "What is the longest film?"
- "What is the average rental rate?"
- "Which actor has appeared in the most films?"
```

### 🧠 Complex Analysis
```
- "Film category popularity analysis"
- "Rental revenue trend analysis"
- "Actor-film relationship network analysis"
- "Customer behavior analysis"
```

## 📈 Performance and Statistics

### ✅ Test Results (Latest Run)
- **Overall Success Rate**: 65% (13/20 tests)
- **Basic SQL Queries**: 62.5% (5/8)
- **Complex Queries**: 40% (2/5)
- **Error Handling**: 75% (3/4)
- **Answer Generation**: 100% (3/3)

### 🚀 Strengths
- ✅ Film counting and filtering
- ✅ Actor/staff queries
- ✅ Average calculations
- ✅ Basic join operations
- ✅ Content search

### 🔄 Areas for Improvement
- 🔄 Complex table relationships
- 🔄 Advanced aggregation operations
- 🔄 Schema relationship understanding

## 🏗️ Technical Architecture

### 🧠 AI Components
- **Chat Model**: Mistral 7B Instruct (SQL generation)
- **Embedding Model**: MXBai Embed Large (document search)
- **Platform**: Ollama (local execution)

### 🗄️ Database
- **DBMS**: PostgreSQL
- **Sample Data**: Pagila (DVD rental system)
- **Tables**: 15+ tables, film/actor/customer/rental data model

### 🔧 Backend Stack
- **Core Logic**: Python AsyncIO
- **Web Framework**: Streamlit + Flask
- **Database Connector**: psycopg2
- **HTTP Client**: aiohttp
- **Visualization**: Plotly + Pandas

### 🎨 Frontend Features
- **Responsive Design**: Modern CSS Grid/Flexbox
- **Real-time Updates**: WebSocket-like updates
- **Interactive Charts**: Plotly.js integration
- **Multi-language**: Turkish/English support

## 🛠️ Installation and Configuration

### � Prerequisites
- **Docker & Docker Compose** (for PostgreSQL)
- **Python 3.8+** with pip
- **Ollama** (local AI models)
- **Git** (for cloning repositories)

### 1. 📥 Clone the Project
```bash
# Clone this repository
git clone https://github.com/aliosmangurbilek/iga_staj_project.git
cd iga_staj_project

# The pagila directory is already included as a submodule
```

### 2. 🗄️ Database Setup (PostgreSQL + Pagila)
```bash
# Navigate to pagila directory
cd pagila/

# Start PostgreSQL with Docker Compose
docker-compose up -d

# Wait for database to be ready (about 30 seconds)
sleep 30

# Verify database is running
docker ps | grep postgres

# Test database connection
psql -h localhost -U postgres -d pagila -c "SELECT COUNT(*) FROM film;"
# Should return: count: 1000
```

**Database Details:**
- **Host**: localhost
- **Port**: 5432
- **Database**: pagila
- **Username**: postgres
- **Password**: 2336

**pgAdmin Web Interface:**
- **URL**: http://localhost:5050
- **Email**: admin@pagila.com
- **Password**: admin2336

### 3. 🤖 Ollama Setup
```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve &

# Download required models
ollama pull mistral:7b-instruct
ollama pull mxbai-embed-large

# Verify models are downloaded
ollama list
```

### 4. 🐍 Python Environment Setup
```bash
# Return to project root
cd ..

# Install required packages
pip install streamlit flask flask-cors plotly pandas psycopg2-binary aiohttp

# Or use requirements.txt (create if needed)
pip install -r requirements.txt
```

### 5. 🔧 Environment Configuration
```bash
# Set environment variables (optional, defaults work)
export DATABASE_URL="postgresql://postgres:2336@localhost:5432/pagila"
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_MODEL="mistral:7b-instruct"
export OLLAMA_EMBEDDING_MODEL="mxbai-embed-large"
```

### 6. 🧪 Test Installation
```bash
# Test your setup
python test_your_setup.py

# Run comprehensive tests
python final_test.py
```

### 7. 🚀 Launch Applications

#### Quick Start (All Services)
```bash
# Start everything with one command
./start_all.sh
```

#### Manual Start (Individual Services)
```bash
# Option 1: Streamlit Pro (Main Interface)
streamlit run app_pro.py --server.port 8502

# Option 2: Flask API + Web Interface
python flask_api.py
```

### 🐳 Docker Compose Details
The `pagila/docker-compose.yml` contains:
```yaml
version: '3.8'
services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: pagila
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: 2336
      POSTGRES_INITDB_ARGS: "--encoding=UTF8"
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./pagila-schema.sql:/docker-entrypoint-initdb.d/01-schema.sql
      - ./pagila-data.sql:/docker-entrypoint-initdb.d/02-data.sql
    command: postgres -c shared_preload_libraries=pg_stat_statements

volumes:
  postgres_data:
```

### ⚙️ Manual Configuration (Optional)
```python
# config.py (for custom settings)
DATABASE_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'pagila',
    'user': 'postgres',
    'password': '2336'
}

OLLAMA_CONFIG = {
    'base_url': 'http://localhost:11434',
    'chat_model': 'mistral:7b-instruct',
    'embed_model': 'mxbai-embed-large',
    'timeout': 300  # 5 minutes
}
```

## 🔄 Troubleshooting

### Common Issues & Solutions

#### 🔍 **Database Connection Issues**
```bash
# Check database is running
docker ps | grep postgres

# Test connection
psql -h localhost -U postgres -d pagila
```

#### 🤖 **Ollama Model Issues**
```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Download missing models
ollama pull mistral:7b-instruct
ollama pull mxbai-embed-large
```

#### 🌐 **Port Conflicts**
```bash
# Check if ports are in use
netstat -tulpn | grep -E "(8501|8502|5000)"

# Kill conflicting processes
sudo kill -9 $(lsof -t -i:8501)
```

### Performance Optimization

#### ⚡ **Query Performance**
- Use specific queries rather than broad analysis
- SQL queries are faster than AI analysis
- Cache frequently used results

#### 🚀 **Model Performance**
- Ensure enough RAM (8GB+ recommended)
- Use GPU acceleration if available
- Consider model quantization for faster inference

## 🔄 Development Roadmap

### 🚧 Planned Features
- [ ] Query caching and optimization
- [ ] Advanced visualization templates
- [ ] Custom model fine-tuning
- [ ] Multi-database support
- [ ] Enhanced error recovery
- [ ] Real-time streaming responses

### 🎯 Technical Improvements
- [ ] Better schema relationship mapping
- [ ] Advanced SQL generation algorithms
- [ ] Improved embedding search
- [ ] Performance monitoring dashboard
- [ ] Auto-scaling capabilities

## 📚 Additional Documentation

- **API Reference**: Check `/docs` endpoint when Flask is running
- **Database Schema**: See `pagila/pagila-schema-diagram.png`
- **Test Coverage**: Run `python comprehensive_test.py` for detailed test reports
- **Performance Benchmarks**: Available in Streamlit Pro dashboard

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Pagila Database**: Sample database for learning PostgreSQL
- **Ollama**: Local LLM platform for AI inference
- **Streamlit & Flask**: Modern web framework components
- **PostgreSQL**: Robust and reliable database system
- **Python Community**: For excellent async and data science libraries

---

**🌟 Ready to explore your data with AI? Start with any interface above!**

For support or questions, please open an issue on the project repository.
