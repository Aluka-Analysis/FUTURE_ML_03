"""
FastAPI application for Resume Screening System.
Endpoints for screening resumes against job descriptions.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime
import uuid
from typing import List

from .schemas import (
    ScreenRequest, ScreenResponse, ScreenFileRequest,
    RankedCandidate, HealthResponse
)
from .predict import ResumeScreener

# Global screener instance
screener = None


def load_screener():
    """Initialize the resume screener with Sentence Transformer model."""
    global screener
    try:
        screener = ResumeScreener(model_path='models/sentence_transformer_model')
        print("✅ Resume screener loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading screener: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup
    print("Starting up...")
    load_screener()
    yield
    # Shutdown
    print("Shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="Resume Screening API",
    description="AI-powered resume screening using semantic similarity and skill extraction. Supports PDF, DOCX, and TXT files.",
    version="2.0.0",
    lifespan=lifespan,
    contact={
        "name": "Aluka Precious Oluchukwu",
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Resume Screening API",
        "version": "2.0.0",
        "status": "operational" if screener and screener.is_loaded else "degraded",
        "endpoints": {
            "/screen": "POST - Screen resumes (text input)",
            "/screen-files": "POST - Screen resumes from files (PDF, DOCX, TXT)",
            "/health": "GET - Health check",
            "/docs": "GET - Interactive API documentation"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    if screener and screener.is_loaded:
        return HealthResponse(status="healthy", models_loaded=True)
    return HealthResponse(status="unhealthy", models_loaded=False)


@app.post("/screen", response_model=ScreenResponse, tags=["Prediction"])
async def screen_resumes(request: ScreenRequest):
    """
    Screen resumes against a job description using text input.
    
    Args:
        request: ScreenRequest with job_description and list of resumes
        
    Returns:
        ScreenResponse with ranked candidates
    """
    if not screener or not screener.is_loaded:
        raise HTTPException(status_code=503, detail="Screener not loaded. Please try again later.")
    
    if not request.job_description or len(request.job_description.strip()) < 10:
        raise HTTPException(status_code=400, detail="Job description too short (minimum 10 characters)")
    
    if not request.resumes:
        raise HTTPException(status_code=400, detail="At least one resume is required")
    
    try:
        # Convert resumes to list of dicts
        resumes = [
            {
                'candidate_id': r.candidate_id,
                'name': r.name,
                'text': r.text
            }
            for r in request.resumes
        ]
        
        # Rank candidates
        ranked = screener.rank_candidates(request.job_description, resumes)
        
        # Build response
        return ScreenResponse(
            job_reference=f"JOB-{str(uuid.uuid4())[:8].upper()}",
            assessment_date=datetime.now().isoformat(),
            total_candidates=len(ranked),
            ranked_candidates=[
                RankedCandidate(
                    rank=r['rank'],
                    candidate_id=r['candidate_id'],
                    name=r.get('name'),
                    final_score=r['final_score'],
                    skill_score=r['skill_score'],
                    semantic_score=r['semantic_score'],
                    matched_skills=r['matched_skills'],
                    missing_skills=r['missing_skills'],
                    recommendation=r['recommendation']
                )
                for r in ranked
            ]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Screening failed: {str(e)}")


@app.post("/screen-files", response_model=ScreenResponse, tags=["Prediction"])
async def screen_resumes_from_files(request: ScreenFileRequest):
    """
    Screen resumes against a job description using file uploads (PDF, DOCX, TXT).
    
    Args:
        request: ScreenFileRequest with job_description and list of resume files
        
    Returns:
        ScreenResponse with ranked candidates
    """
    if not screener or not screener.is_loaded:
        raise HTTPException(status_code=503, detail="Screener not loaded. Please try again later.")
    
    if not request.job_description or len(request.job_description.strip()) < 10:
        raise HTTPException(status_code=400, detail="Job description too short (minimum 10 characters)")
    
    if not request.resumes:
        raise HTTPException(status_code=400, detail="At least one resume is required")
    
    try:
        # Convert files to list of dicts
        resume_files = [
            {
                'candidate_id': r.candidate_id,
                'name': r.name,
                'filename': r.filename,
                'content_base64': r.content_base64
            }
            for r in request.resumes
        ]
        
        # Rank candidates from files
        ranked = screener.rank_candidates_from_files(request.job_description, resume_files)
        
        # Build response
        return ScreenResponse(
            job_reference=f"JOB-{str(uuid.uuid4())[:8].upper()}",
            assessment_date=datetime.now().isoformat(),
            total_candidates=len(ranked),
            ranked_candidates=[
                RankedCandidate(
                    rank=r['rank'],
                    candidate_id=r['candidate_id'],
                    name=r.get('name'),
                    final_score=r['final_score'],
                    skill_score=r['skill_score'],
                    semantic_score=r['semantic_score'],
                    matched_skills=r['matched_skills'],
                    missing_skills=r['missing_skills'],
                    recommendation=r['recommendation']
                )
                for r in ranked
            ]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Screening failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)