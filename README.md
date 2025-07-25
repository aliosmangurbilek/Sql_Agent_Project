# 🎬 Pagila AI Assistant Pro

Advanced AI-powered SQL query assistant that converts natural language questions into SQL queries and provides intelligent database analysis for the Pagila sample database.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/aliosmangurbilek/iga_staj_project.git
cd iga_staj_project

# Install dependencies
pip install -r config/requirements.txt

# Set up environment variables
cp config/.env.example config/.env
# Edit config/.env with your API keys

# Start the application
python working_app.py
# OR
streamlit run working_app.py
```

The app will be available at: http://localhost:8501

## 🎯 Core Features

### 🤖 Natural Language to SQL
- **Ask questions in plain English**: "How many movies are in the database?"
- **Smart SQL generation**: Converts natural language to optimized PostgreSQL queries
- **Context-aware responses**: Uses database schema for accurate query generation
- **Safety checks**: Prevents dangerous operations (DROP, DELETE, etc.)

### 🎨 Multi-Provider AI Support
- **Ollama** (Local, Free): mistral:7b-instruct, llama3.1:8b, codellama:7b, qwen2.5-coder:7b
- **OpenAI**: gpt-4o-mini, gpt-4o, gpt-4-turbo, gpt-4, gpt-3.5-turbo
- **Google Gemini**: gemini-1.5-pro, gemini-1.5-flash, gemini-pro, gemini-pro-vision
- **OpenRouter**: Access to multiple models including free options (deepseek, llama, phi-3)

### 📊 Advanced Visualization
- **Auto-visualization**: Smart chart generation based on query results
- **Multiple chart types**: Bar charts, histograms, scatter plots, metrics
- **Interactive plots**: Powered by Plotly for rich user interaction
- **Data export**: Download results as CSV or JSON

### 🛠️ Professional Features
- **Real-time query execution**: See results instantly
- **Error handling**: Comprehensive timeout and connection error management
- **Model selection**: Dynamic dropdown menus for each AI provider
- **Database schema integration**: Full Pagila database schema awareness
- **Performance monitoring**: Query execution time tracking

## 🏗️ Architecture

### Project Structure
```
├── working_app.py              # Main Streamlit application
├── src/                        # Modular components
│   ├── components/            # Reusable UI components
│   ├── pages/                 # Application pages
│   ├── utils/                 # Utility functions
│   └── schema_tools.py        # Database interaction tools
├── config/
│   ├── requirements.txt       # Python dependencies
│   └── .env                   # Environment configuration
├── data/pagila/               # Sample database files
├── docs/                      # Documentation
└── scripts/                   # Setup and utility scripts
```

### Key Components
- **Natural Language Processing**: AI-powered query understanding
- **SQL Generation**: Context-aware SQL query creation
- **Database Integration**: PostgreSQL connection and schema management
- **Visualization Engine**: Automatic chart generation
- **Multi-Provider Support**: Flexible AI backend switching

## 🔧 Configuration

### Environment Variables
Create `config/.env` with your API keys:

```env
# AI Provider API Keys
OLLAMA_API_URL=http://localhost:11434
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_gemini_key_here
OPENROUTER_API_KEY=your_openrouter_key_here

# Default Models
OLLAMA_MODEL=mistral:7b-instruct
OPENAI_MODEL=gpt-4o-mini
GEMINI_MODEL=gemini-pro
OPENROUTER_MODEL=deepseek/deepseek-chat-v3-0324:free

# Database Configuration
DATABASE_URL=postgresql://your-user-name:your-password@localhost:5432/pagila
```

### Database Setup
1. **Install PostgreSQL**
2. **Load Pagila sample database**:
   ```bash
   cd data/pagila
   # Follow instructions in pagila/README.md
   ```
3. **Update DATABASE_URL** in your `.env` file

## 📚 Usage Examples

### Natural Language Queries
```
"How many movies are there?"
→ SELECT COUNT(*) FROM film;

"Top 5 longest movies"
→ SELECT title, length FROM film ORDER BY length DESC LIMIT 5;

"Movies with Tom Cruise"
→ SELECT f.title FROM film f 
  JOIN film_actor fa ON f.film_id = fa.film_id 
  JOIN actor a ON fa.actor_id = a.actor_id 
  WHERE a.first_name = 'TOM' AND a.last_name = 'CRUISE';
```

### Provider Switching
1. Select AI provider from dropdown (Ollama, OpenAI, Gemini, OpenRouter)
2. Choose specific model from provider's model list
3. Ask your question in natural language
4. Get SQL query and results with visualization

## 🚀 Advanced Features

### Error Handling
- **Timeout Management**: 120-second timeout with clear error messages
- **Connection Recovery**: Automatic retry mechanisms
- **API Key Validation**: Clear guidance for missing credentials
- **SQL Safety**: Prevention of dangerous database operations

### Performance Optimization
- **Query Caching**: Reduced redundant database calls
- **Result Limiting**: Automatic pagination for large datasets
- **Efficient Visualization**: Optimized chart rendering
- **Memory Management**: Smart data handling for large results

### User Experience
- **Responsive Design**: Works on desktop and mobile
- **Real-time Feedback**: Progress indicators and status updates
- **Export Options**: Multiple data download formats
- **Template Queries**: Pre-built example questions

## 📖 Documentation

- [Ollama Setup Guide](docs/OLLAMA_SETUP.md) - Local AI setup instructions
- [OpenRouter Integration](docs/README_OPENROUTER.md) - Multi-model API setup
- [Modular Structure](docs/MODULAR_STRUCTURE.md) - Architecture deep dive

## 🛠️ Development

### Requirements
- Python 3.8+
- PostgreSQL 12+
- Streamlit 1.28+
- Required Python packages (see `config/requirements.txt`)

### Local Development
```bash
# Install development dependencies
pip install -r config/requirements.txt

# Run with auto-reload
streamlit run working_app.py --server.runOnSave true

# Database setup
python scripts/setup_document_search.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Pagila Database**: Sample DVD rental database for PostgreSQL
- **Streamlit**: Amazing framework for data apps
- **OpenAI, Google, Meta**: AI model providers
- **Ollama**: Local AI inference platform

---

**Built with Streamlit, PostgreSQL, and ❤️**

*Transform your database questions into insights with the power of AI*