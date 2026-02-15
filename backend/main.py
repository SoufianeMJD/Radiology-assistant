"""
FastAPI Main Application - Multimodal Radiological AI Assistant
Orchestrates Vision, RAG, and LLM services for medical image analysis
"""
import os
import uuid
import traceback
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from config import API_CONFIG
from services import vision_service, rag_service, llm_service


# Initialize FastAPI app
app = FastAPI(
    title="Multimodal Radiological AI Assistant",
    description="Medical image analysis with Vision, RAG, and LLM",
    version="1.0.0"
)

# CORS configuration for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# Pydantic models
class AnalysisResponse(BaseModel):
    """Response model for image analysis"""
    success: bool
    findings: List[Dict]
    detected_count: int
    similar_cases: List[Dict]
    generated_report: str
    processing_time: float
    error: Optional[str] = None


class ChatMessage(BaseModel):
    """Chat message model"""
    role: str
    content: str


class ChatRequest(BaseModel):
    """Chat request model"""
    history: List[Dict]
    message: str
    case_context: Optional[str] = ""


class ChatResponse(BaseModel):
    """Chat response model"""
    success: bool
    response: str
    error: Optional[str] = None


# Global state for current case
current_case = {
    "findings": [],
    "report": "",
    "similar_cases": []
}


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("\n" + "="*60)
    print("MULTIMODAL RADIOLOGICAL AI ASSISTANT")
    print("="*60)
    
    try:
        # Load vision model
        print("\n[STARTUP] Initializing Vision Service...")
        vision_service.load_model()
        
        # Check if RAG vector store exists
        print("\n[STARTUP] Initializing RAG Service...")
        if not rag_service.load_index():
            print("\n" + "⚠"*30)
            print("WARNING: Vector database not found!")
            print("Please run: python scripts/build_vector_db.py")
            print("⚠"*30 + "\n")
        else:
            print("[STARTUP] ✓ RAG vector store loaded successfully")
        
        # Note: LLM is loaded on-demand to save memory
        print("\n[STARTUP] LLM will be loaded on first request")
        
        print("\n" + "="*60)
        print("✓ Services initialized")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n[STARTUP ERROR] {str(e)}")
        print(traceback.format_exc())


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "Multimodal Radiological AI Assistant",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/analyze",
            "chat": "/chat",
            "status": "/status"
        }
    }


@app.get("/status")
async def get_status():
    """Get service status"""
    return {
        "vision_model_loaded": vision_service.model is not None,
        "rag_index_loaded": rag_service.index is not None,
        "llm_model_loaded": llm_service.model is not None,
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze chest X-ray image
    
    Process:
    1. Vision: Detect pathologies
    2. RAG: Retrieve similar cases
    3. LLM: Generate professional report
    """
    start_time = datetime.now()
    temp_file_path = None
    
    try:
        # Validate file
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Check file size
        file_content = await file.read()
        if len(file_content) > API_CONFIG["max_file_size"]:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds {API_CONFIG['max_file_size'] / (1024*1024)}MB limit"
            )
        
        # Save temporary file
        temp_filename = f"{uuid.uuid4()}_{file.filename}"
        temp_file_path = UPLOAD_DIR / temp_filename
        
        with open(temp_file_path, "wb") as f:
            f.write(file_content)
        
        print(f"\n[ANALYZE] Processing: {file.filename}")
        print("="*50)
        
        # Step 1: Vision Analysis
        print("[STEP 1/3] Running vision analysis...")
        vision_result = vision_service.predict(str(temp_file_path))
        findings = vision_result["pathologies"]
        
        print(f"✓ Detected {len(findings)} pathologies")
        for finding in findings[:5]:  # Print top 5
            print(f"  - {finding['name']}: {finding['confidence']:.2f}")
        
        # Step 2: RAG Retrieval
        print("\n[STEP 2/3] Retrieving similar cases...")
        
        if findings:
            # Create query from findings
            query = " ".join([f["name"] for f in findings[:3]])
        else:
            query = "Normal chest X-ray"
        
        similar_cases = rag_service.retrieve(query, top_k=3)
        print(f"✓ Retrieved {len(similar_cases)} similar cases")
        
        # Format RAG context
        rag_context = "\n\n".join([
            f"Case {r['rank']}: {r['report'][:300]}..."
            for r in similar_cases
        ])
        
        # Step 3: LLM Report Generation
        print("\n[STEP 3/3] Generating professional report...")
        
        if findings:
            generated_report = llm_service.generate_report(findings, rag_context)
        else:
            generated_report = llm_service.generate_report(
                [{"name": "No significant pathology", "confidence": 1.0}],
                rag_context
            )
        
        print("✓ Report generated")
        
        # Update global case state
        current_case["findings"] = findings
        current_case["report"] = generated_report
        current_case["similar_cases"] = similar_cases
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\n✓ Analysis complete in {processing_time:.2f}s")
        print("="*50 + "\n")
        
        return AnalysisResponse(
            success=True,
            findings=findings,
            detected_count=len(findings),
            similar_cases=similar_cases,
            generated_report=generated_report,
            processing_time=processing_time
        )
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n[ERROR] {error_msg}")
        print(traceback.format_exc())
        
        return AnalysisResponse(
            success=False,
            findings=[],
            detected_count=0,
            similar_cases=[],
            generated_report="",
            processing_time=0,
            error=error_msg
        )
    
    finally:
        # Clean up temporary file
        if temp_file_path and temp_file_path.exists():
            os.remove(temp_file_path)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Handle conversational queries about the current case
    """
    try:
        print(f"\n[CHAT] User: {request.message}")
        
        # Build case context
        case_context = ""
        if current_case["findings"]:
            findings_str = ", ".join([f["name"] for f in current_case["findings"][:5]])
            case_context = f"Current case findings: {findings_str}\n\n"
        
        if current_case["report"]:
            case_context += f"Generated Report:\n{current_case['report'][:500]}..."
        
        # Generate response
        response = llm_service.chat(
            history=request.history,
            user_input=request.message,
            case_context=case_context
        )
        
        print(f"[CHAT] Assistant: {response[:100]}...")
        
        return ChatResponse(
            success=True,
            response=response
        )
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n[CHAT ERROR] {error_msg}")
        print(traceback.format_exc())
        
        return ChatResponse(
            success=False,
            response="",
            error=error_msg
        )





if __name__ == "__main__":
    print("\nStarting Multimodal Radiological AI Assistant...")
    print(f"API will be available at: http://{API_CONFIG['host']}:{API_CONFIG['port']}")
    print("\n")
    
    uvicorn.run(
        app,
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        log_level="info"
    )
