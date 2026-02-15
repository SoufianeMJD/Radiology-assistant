# Gemini API Migration - Setup Guide

## Quick Setup

### 1. Get Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key

### 2. Configure Environment

Navigate to the backend directory and create `.env` file:

```bash
cd backend
copy .env.example .env  # Windows
# or
cp .env.example .env    # Linux/Mac
```

Edit `.env` and add your API key:
```
GEMINI_API_KEY=your_actual_api_key_here
```

### 3. Reinstall Dependencies

```bash
# Uninstall old dependencies (optional but recommended)
pip uninstall transformers accelerate bitsandbytes protobuf tiktoken sentencepiece -y

# Install new dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Test the Setup

```bash
python -c "from services.llm_service import llm_service; llm_service.load_model(); print('✓ Gemini API configured successfully!')"
```

### 5. Run the Application

```bash
python main.py
```

## What Changed

**Removed:**
- MedAlpaca-7B model (~27GB)
- Transformers library
- 4-bit quantization dependencies
- GPU requirements for LLM

**Added:**
- Google Gemini API integration
- Environment variable management (.env)
- Simplified configuration

## Benefits

- **No Model Downloads**: Zero wait time for model downloads
- **Reduced Memory**: From 32GB RAM to ~4GB RAM
- **No GPU Required**: API handles computation
- **Faster Responses**: Gemini 1.5 Flash is optimized for speed
- **Better Quality**: Access to Google's latest models

## API Key Limits

**Free Tier:**
- 60 requests per minute
- 1,500 requests per day
- Sufficient for development and testing

**Paid Tier:**
- Higher rate limits available
- Pay-per-use pricing

## Troubleshooting

**Error: "GEMINI_API_KEY not found"**
- Ensure `.env` file exists in `backend/` directory
- Check API key is correctly copied (no extra spaces)

**Error: "API key not valid"**
- Verify API key in Google AI Studio
- Generate a new key if needed

**Error: "Resource exhausted"**
- Hit rate limit - wait 1 minute and try again
- Consider implementing rate limiting in code
