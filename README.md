# 🏥 Multimodal Radiological AI Assistant

A sophisticated medical imaging analysis system that combines Computer Vision, Retrieval-Augmented Generation (RAG), and Large Language Models to provide professional radiological reports from chest X-rays.

![Architecture](https://img.shields.io/badge/Architecture-Vision%20%2B%20RAG%20%2B%20LLM-blue)
![Backend](https://img.shields.io/badge/Backend-FastAPI%20%2B%20PyTorch-green)
![Frontend](https://img.shields.io/badge/Frontend-React%20%2B%20Tailwind-purple)

## 🎯 Features

### Vision Module
- **TorchXRayVision DenseNet121** for pathology detection
- Automated preprocessing and normalization
- Confidence-based pathology classification

### RAG Module
- **Automatic dataset download** via kagglehub
- **FAISS** vector search for similar case retrieval
- **Sentence Transformers** for semantic embeddings

### LLM Module
- **Medical LLM** (MedAlpaca-7B) with 4-bit quantization
- Professional report generation with structured sections
- Interactive chat assistant for case discussion

### Frontend
- Medical-grade dark mode UI with glassmorphism
- Drag-and-drop X-ray upload
- Advanced image viewer with zoom/pan controls
- Real-time multi-stage analysis progress
- PDF export functionality
- ChatGPT-style medical assistant

---

## 📋 Prerequisites

### System Requirements
- **OS**: Windows 10/11, Linux, or macOS
- **GPU**: NVIDIA GPU with 12GB+ VRAM (recommended)
- **RAM**: 16GB minimum
- **Storage**: ~20GB for models and data

### Software Requirements
- **Python**: 3.10 or 3.11
- **Node.js**: 18+ and npm
- **CUDA**: 11.8+ (for GPU acceleration)

---

## 🚀 Installation

### Step 1: Clone or Navigate to Project

```bash
cd "e:\Desktop Files\SDIA\S3\GenAI\GenAI Project LLM+RAG"
```

### Step 2: Backend Setup

#### Create Virtual Environment

```bash
cd backend
python -m venv venv
```

#### Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

#### Install Dependencies

**IMPORTANT: Choose GPU or CPU Installation**

**For GPU (NVIDIA CUDA) - Recommended for Performance:**

1. **Install PyTorch with CUDA support FIRST:**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

2. **Then install remaining dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Verify CUDA is working:**
   ```bash
   python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
   ```
   You should see: `CUDA Available: True`

**For CPU (No GPU or Testing):**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note**: GPU installation requires Python 3.10 or 3.11 (PyTorch with CUDA doesn't support Python 3.13 yet). CPU installation works with any Python version 3.10+.

### Step 3: Frontend Setup

Open a **new terminal** and navigate to frontend:

```bash
cd frontend
npm install
```

---

## 🎬 Running the Application

### Start Backend Server

In the **backend** terminal (with virtual environment activated):

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Expected output:**
```
[CONFIG] Using device: cuda
[STARTUP] Initializing Vision Service...
[VISION] Loading DenseNet121 model...
[VISION] Model loaded successfully on cuda
[STARTUP] Initializing RAG Service...
✓ All services initialized successfully
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Start Frontend Development Server

In the **frontend** terminal:

```bash
npm run dev
```

**Expected output:**
```
  VITE v5.0.8  ready in 500 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

### Access the Application

Open your browser and navigate to:
```
http://localhost:5173
```

---

## 📊 Dataset Configuration

The system automatically downloads the **Indiana University Chest X-rays** dataset on first use via kagglehub.

### First-Time RAG Initialization

On first run, the RAG index will be created automatically when you analyze your first image. Alternatively, you can pre-initialize it:

1. Visit `http://localhost:8000/init-rag` in your browser, or
2. Use curl:
   ```bash
   curl -X POST http://localhost:8000/init-rag
   ```

This will:
- Download the Indiana dataset (~500MB)
- Process medical reports
- Generate embeddings
- Build FAISS index

**This process takes 5-10 minutes** but only needs to be done once.

---

## 🏗️ Project Structure

```
GenAI Project LLM+RAG/
├── backend/
│   ├── services/
│   │   ├── __init__.py
│   │   ├── vision_service.py    # TorchXRayVision pathology detection
│   │   ├── rag_service.py        # FAISS + SentenceTransformers
│   │   └── llm_service.py        # 4-bit quantized medical LLM
│   ├── config.py                 # Centralized configuration
│   ├── main.py                   # FastAPI application
│   ├── requirements.txt
│   ├── data/                     # Auto-created for datasets
│   │   └── faiss_index/          # FAISS index storage
│   ├── models/                   # Model cache
│   └── uploads/                  # Temporary image storage
│
└── frontend/
    ├── src/
    │   ├── components/
    │   │   ├── ImageViewer.jsx   # Drag-drop + zoom/pan
    │   │   ├── LoadingStates.jsx # Multi-stage progress
    │   │   ├── ReportTab.jsx     # Markdown report + PDF export
    │   │   └── ChatTab.jsx       # Conversational assistant
    │   ├── services/
    │   │   └── api.js            # Axios API client
    │   ├── App.jsx               # Main application
    │   ├── main.jsx
    │   └── index.css             # Medical-grade theme
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── tailwind.config.js
```

---

## 🎨 Usage Guide

### 1. Upload X-Ray Image
- Drag and drop an X-ray image onto the left panel
- Or click to browse and select a file
- Supported formats: JPEG, PNG (max 10MB)

### 2. Automated Analysis
The system performs three stages:
1. **Vision Analysis**: Detects pathologies using DenseNet121
2. **Similar Cases**: Retrieves relevant historical reports via RAG
3. **Report Generation**: Creates professional medical report using LLM

### 3. Review Report
- View detected pathologies with confidence scores
- Read AI-generated structured report
- Explore similar historical cases
- Export report as PDF

### 4. Ask Questions
- Switch to the "Assistant" tab
- Ask questions about the findings
- Get medically-informed responses
- Maintain conversation context

---

## 🔧 Configuration

### Backend Configuration (`backend/config.py`)

```python
MODEL_CONFIG = {
    "vision": {
        "confidence_threshold": 0.5,  # Adjust pathology detection threshold
    },
    "rag": {
        "top_k": 3,  # Number of similar cases to retrieve
    },
    "llm": {
        "model_name": "medalpaca/medalpaca-7b",  # Change LLM model
        "max_new_tokens": 512,
        "temperature": 0.7,
    }
}
```

### Alternative Medical LLMs

You can switch to different medical models by editing `config.py`:

```python
"llm": {
    "model_name": "epfl-llm/meditron-7b",  # Alternative option
    # OR
    "model_name": "axiong/PMC_LLaMA_13B",  # Requires more VRAM
}
```

---

## 🐛 Troubleshooting

### GPU Out of Memory

**Error**: `CUDA out of memory`

**Solutions**:
1. Close other GPU-intensive applications
2. Reduce batch size in configuration
3. Use a smaller LLM model
4. System will automatically fallback to CPU (slower)

### Dataset Download Issues

**Error**: `kagglehub authentication failed`

**Solution**:
1. Ensure you have a Kaggle account
2. Create API credentials at https://www.kaggle.com/settings
3. Place `kaggle.json` in:
   - Windows: `C:\Users\<username>\.kaggle\`
   - Linux/Mac: `~/.kaggle/`

### Port Already in Use

**Error**: `Address already in use`

**Solution**:
```bash
# Change port in backend/config.py
API_CONFIG = {
    "port": 8001,  # Use different port
}

# And update frontend/src/services/api.js
const API_BASE_URL = 'http://localhost:8001';
```

### Model Download Slow

Models are downloaded from HuggingFace. First-time setup requires:
- TorchXRayVision: ~50MB
- SentenceTransformers: ~80MB
- MedAlpaca-7B (4-bit): ~4GB

**Solution**: Be patient or use a faster internet connection. Models are cached after first download.

---

## 📊 API Endpoints

### GET `/`
Health check endpoint

### GET `/status`
Returns service status and loaded models

### POST `/analyze`
Analyze chest X-ray image
- **Input**: Multipart form with image file
- **Output**: JSON with findings, report, and similar cases

### POST `/chat`
Handle conversational queries
- **Input**: JSON with message history and current message
- **Output**: JSON with assistant response

### POST `/init-rag`
Manually initialize RAG index
- **Output**: JSON with success status and report count

---

## 🔒 Important Notes

### Medical Disclaimer

> ⚠️ **This system is for research and educational purposes only.**  
> - NOT intended for clinical diagnosis
> - NOT a replacement for professional medical advice
> - Always consult qualified healthcare professionals

### Performance Expectations

- **First analysis**: 30-60 seconds (model loading + inference)
- **Subsequent analyses**: 10-20 seconds
- **Chat responses**: 3-5 seconds

### GPU vs CPU

| Component | GPU (CUDA) | CPU |
|-----------|-----------|-----|
| Vision Model | 0.5s | 2-3s |
| RAG Search | 0.1s | 0.3s |
| LLM Generation | 5-10s | 30-60s |

---

## 🚀 Production Deployment

For production use:

1. **Build Frontend**:
   ```bash
   cd frontend
   npm run build
   ```

2. **Serve Static Files**: Configure FastAPI to serve the built frontend

3. **Use Production Server**:
   ```bash
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
   ```

4. **Enable HTTPS**: Use nginx as reverse proxy with SSL

5. **Set Environment Variables**: Store sensitive config in `.env`

---

## 📈 Future Enhancements

- [ ] Multi-modal image support (CT, MRI)
- [ ] User authentication and case history
- [ ] Advanced visualization tools (heatmaps, segmentation)
- [ ] Integration with PACS systems
- [ ] Fine-tuning on custom datasets
- [ ] Multi-language support

---

## 🤝 Contributing

This is a research/educational project. Contributions are welcome:

1. Fork the repository
2. Create a feature branch
3. Implement your enhancement
4. Submit a pull request

---

## 📄 License

This project uses multiple open-source libraries and models:
- TorchXRayVision: Apache 2.0
- FAISS: MIT License
- Transformers: Apache 2.0
- React: MIT License

---

## 🙏 Acknowledgments

- **TorchXRayVision** by mlmed
- **Indiana University** for the chest X-ray dataset
- **HuggingFace** for model hosting
- **MedAlpaca Team** for the medical LLM

---

## 📞 Support

For issues or questions:
- Check the troubleshooting section
- Review console logs for detailed error messages
- Ensure all dependencies are correctly installed

**Developed with ❤️ for advancing medical AI research**
