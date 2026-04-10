"""
Pydantic schemas for request and response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import uuid


class ResumeInput(BaseModel):
    """Single resume input (text/copy-paste)."""
    candidate_id: str = Field(..., description="Unique identifier for the candidate")
    name: Optional[str] = Field(None, description="Candidate's name (optional)")
    text: str = Field(..., min_length=10, description="Full resume text")


class ResumeFileInput(BaseModel):
    """Resume input from file upload (PDF, DOCX, TXT)."""
    candidate_id: str = Field(..., description="Unique identifier for the candidate")
    name: Optional[str] = Field(None, description="Candidate's name (optional)")
    filename: str = Field(..., description="Original filename with extension")
    content_base64: str = Field(..., description="Base64 encoded file content")


class ScreenRequest(BaseModel):
    """Request schema for screening endpoint."""
    job_description: str = Field(..., min_length=10, description="Job description text")
    resumes: List[ResumeInput] = Field(..., description="List of resumes to screen (text/copy-paste)")


class ScreenFileRequest(BaseModel):
    """Request schema for screening with file uploads."""
    job_description: str = Field(..., min_length=10, description="Job description text")
    resumes: List[ResumeFileInput] = Field(..., description="List of resume files to screen")


class RankedCandidate(BaseModel):
    """Ranked candidate output."""
    rank: int
    candidate_id: str
    name: Optional[str]
    final_score: float
    skill_score: float
    semantic_score: float
    matched_skills: List[str]
    missing_skills: List[str]
    recommendation: str


class ScreenResponse(BaseModel):
    """Response schema for screening endpoint."""
    job_reference: str
    assessment_date: str
    total_candidates: int
    ranked_candidates: List[RankedCandidate]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: bool