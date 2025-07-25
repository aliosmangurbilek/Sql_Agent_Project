# OpenRouter Integration Guide

## What is OpenRouter?

OpenRouter is a unified API that provides access to multiple AI models from different providers including OpenAI, Anthropic, Google, Meta, and more. It's particularly useful because:

- **Multiple Models**: Access to GPT-4, Claude, Gemini, Llama, and many others through one API
- **Free Tier**: Some models like `meta-llama/llama-3.1-8b-instruct:free` are completely free
- **Cost-Effective**: Competitive pricing compared to direct API access
- **No Rate Limits**: On most models
- **Easy Switching**: Change models without changing your code

## Setup Instructions

### 1. Get an OpenRouter API Key

1. Visit [https://openrouter.ai/keys](https://openrouter.ai/keys)
2. Sign up for a free account
3. Create a new API key
4. Copy the key (it starts with `sk-or-v1-`)

### 2. Configure Your Environment

Add these settings to your `.env` file:

```env
# OpenRouter Configuration
AI_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here
OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct:free
OPENROUTER_SITE_URL=http://localhost:8501
OPENROUTER_APP_NAME=IGA_Staj_Project
```

### 3. Test Your Setup

Run the test script to verify everything works:

```bash
python test_openrouter_setup.py
```

## Available Models

### Free Models
- `meta-llama/llama-3.1-8b-instruct:free` - Fast, capable, completely free
- `microsoft/phi-3-mini-128k-instruct:free` - Compact but powerful

### Premium Models
- `openai/gpt-4-turbo` - OpenAI's latest GPT-4
- `anthropic/claude-3.5-sonnet` - Anthropic's Claude 3.5
- `google/gemini-pro-1.5` - Google's Gemini Pro
- `microsoft/wizardlm-2-8x22b` - Microsoft's WizardLM

### Specialized Models
- `meta-llama/codellama-34b-instruct` - Code generation specialist
- `anthropic/claude-3-haiku` - Fast, cheap for simple tasks
- `openai/gpt-3.5-turbo` - Reliable and cost-effective

## Usage in the Application

### Via Streamlit UI

1. Open the app: `streamlit run app_pro.py`
2. Go to the sidebar "🔧 Change AI Provider"
3. Select "OpenRouter (API Key Required)"
4. Enter your API key and choose a model
5. Click "🔄 Switch to OpenRouter"

### Via Environment Variables

Set `AI_PROVIDER=openrouter` in your `.env` file and restart the app.

### Programmatically

```python
import os
from schema_tools import ask_db

# Set provider
os.environ['AI_PROVIDER'] = 'openrouter'
os.environ['OPENROUTER_API_KEY'] = 'your-key'
os.environ['OPENROUTER_MODEL'] = 'meta-llama/llama-3.1-8b-instruct:free'

# Use it
result = await ask_db("How many films are in the database?")
```

## Model Recommendations

### For SQL Generation
- **Best Free**: `meta-llama/llama-3.1-8b-instruct:free`
- **Best Premium**: `anthropic/claude-3.5-sonnet`
- **Most Reliable**: `openai/gpt-4-turbo`

### For Code Tasks
- **Specialized**: `meta-llama/codellama-34b-instruct`
- **General**: `openai/gpt-4-turbo`

### For Fast Responses
- **Ultra Fast**: `anthropic/claude-3-haiku`
- **Free & Fast**: `meta-llama/llama-3.1-8b-instruct:free`

## Cost Comparison

| Model | Provider | Cost per 1M tokens | Notes |
|-------|----------|-------------------|-------|
| Llama 3.1 8B | OpenRouter | FREE | Completely free |
| GPT-3.5 Turbo | OpenRouter | $0.50 | Cheaper than OpenAI direct |
| GPT-4 Turbo | OpenRouter | $10.00 | Same as OpenAI direct |
| Claude 3.5 Sonnet | OpenRouter | $3.00 | Cheaper than Anthropic direct |

## Troubleshooting

### Common Issues

1. **401 Unauthorized**: Check your API key is correct
2. **Model not found**: Verify the model name is exact
3. **Rate limits**: Try a different model or wait
4. **No response**: Check your internet connection

### Error Messages

- `No auth credentials found`: API key missing or invalid
- `Model not supported`: Try a different model from the list above
- `Insufficient credits`: Add credits to your OpenRouter account

## Advanced Configuration

### Custom Headers

OpenRouter allows custom headers for tracking:

```env
OPENROUTER_SITE_URL=https://yourdomain.com
OPENROUTER_APP_NAME=Your_App_Name
```

### Model Parameters

Some models support additional parameters:

```python
# In the API call (advanced users)
{
    "temperature": 0.0,      # Deterministic responses
    "max_tokens": 1000,      # Limit response length
    "top_p": 0.9,           # Nucleus sampling
    "presence_penalty": 0.1  # Reduce repetition
}
```

## Integration Benefits

1. **Fallback Strategy**: If OpenAI is down, switch to Claude instantly
2. **Cost Optimization**: Use free models for development, premium for production
3. **Model Comparison**: Test different models with the same prompts
4. **Future-Proof**: Access to new models as they're released

## Security Notes

- Never commit API keys to git
- Use environment variables or `.env` files
- Rotate keys regularly
- Monitor usage on the OpenRouter dashboard

For more information, visit [OpenRouter Documentation](https://openrouter.ai/docs)
