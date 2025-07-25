# 🦙 Ollama Setup Guide

Ollama is a free, local AI solution that runs on your computer. Perfect for development and avoiding API costs!

## 📥 Installation

### Linux/WSL:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### macOS:
```bash
brew install ollama
```

### Windows:
Download from: https://ollama.ai/download

## 🚀 Quick Start

1. **Start Ollama service:**
```bash
ollama serve
```

2. **Pull the required model:**
```bash
ollama pull mistral:7b-instruct
```

3. **Optional - Pull embedding model:**
```bash
ollama pull mxbai-embed-large
```

## 🔧 Configuration

Your `.env` file is already configured for Ollama:
```properties
AI_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=mistral:7b-instruct
OLLAMA_EMBEDDING_MODEL=mxbai-embed-large
```

## 📋 Available Models

### Recommended for SQL tasks:
- `mistral:7b-instruct` (Default, good balance)
- `llama3.1:8b` (Latest Llama model)
- `codellama:7b` (Code-focused)

### Lightweight options:
- `phi3:mini` (3.8B parameters, very fast)
- `qwen2.5:7b` (Good for structured tasks)

### To change model:
```bash
ollama pull <model-name>
```
Then update `OLLAMA_MODEL` in `config/.env`

## 🔍 Testing

```bash
# Test if Ollama is running
curl http://localhost:11434/api/tags

# Test the model
ollama run mistral:7b-instruct "Hello, can you help with SQL?"
```

## 💡 Tips

- **Memory**: 7B models need ~8GB RAM
- **Speed**: Smaller models = faster responses
- **Cost**: Completely free, runs locally
- **Privacy**: No data sent to external APIs

## 🚀 Run the App

Once Ollama is set up:
```bash
python run_app.py
```

The app will automatically use Ollama for AI features!

---

**Need help?** 
- Ollama docs: https://ollama.ai/
- Model library: https://ollama.ai/library
