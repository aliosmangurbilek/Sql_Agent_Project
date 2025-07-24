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
```bash
# Basic Streamlit App
streamlit run streamlit_app.py --port 8501

# Advanced Pro App (Recommended)
streamlit run app_pro.py --port 8502

# Flask API + Web Interface
python flask_api.py
```

### 4. Access Your AI Assistant
- **🎨 Streamlit Pro**: http://localhost:8502 *(Recommended for interactive analysis)*
- **🔌 Flask API**: http://localhost:5000 *(For REST API and web interface)*
- **💻 Basic Streamlit**: http://localhost:8501 *(Simple query interface)*

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

| Feature | Streamlit Basic | Streamlit Pro | Flask API | CLI |
|---------|-----------------|---------------|-----------|-----|
| 🖥️ Web Interface | ✅ | ✅ | ✅ | ❌ |
| 🔌 REST API | ❌ | ❌ | ✅ | ❌ |
| 📊 Visualization | ✅ | ✅✅ | ❌ | ❌ |
| 📈 Dashboard | ❌ | ✅ | ❌ | ❌ |
| 📋 Query History | ❌ | ✅ | ❌ | ❌ |
| ⚡ Performance Metrics | ❌ | ✅ | ❌ | ❌ |
| 🔧 System Tools | ❌ | ✅ | ❌ | ❌ |
| 💾 Data Download | ✅ | ✅ | ❌ | ❌ |
| 🤖 AI Analysis | ✅ | ✅ | ✅ | ✅ |
| 🚀 Fast SQL | ✅ | ✅ | ✅ | ✅ |

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

### 🐳 Quick Start with Docker
```bash
# Start Pagila database
cd pagila/
docker-compose up -d

# Python environment
pip install -r requirements.txt

# Start all interfaces
./start_all.sh  # (Optional script)
```

### ⚙️ Manual Configuration
```python
# config.py (recommended)
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
