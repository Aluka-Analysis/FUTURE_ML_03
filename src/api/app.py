from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import base64
import io
import json

# PDF and DOCX support
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    import docx
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load Sentence Transformer model
print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded")

# Default skill patterns (fallback if frontend doesn't send skills)
DEFAULT_SKILL_PATTERNS = {
    'python': r'\bpython\b',
    'machine learning': r'\b(machine learning|ml)\b',
    'tensorflow': r'\btensorflow\b',
    'pytorch': r'\bpytorch\b',
    'sql': r'\bsql\b',
    'data analysis': r'\bdata analysis\b',
    'excel': r'\bexcel\b',
    'tableau': r'\btableau\b',
    'aws': r'\baws\b',
    'docker': r'\bdocker\b',
    'kubernetes': r'\bkubernetes\b',
    'communication': r'\bcommunication\b',
    'leadership': r'\bleadership\b'
}

DEFAULT_SKILL_WEIGHTS = {
    'python': 3, 'machine learning': 3, 'tensorflow': 3, 'pytorch': 3,
    'sql': 2, 'data analysis': 2, 'tableau': 2, 'aws': 2, 'docker': 2, 'kubernetes': 2,
    'excel': 1, 'communication': 1, 'leadership': 1
}

# ============================================================
# PDF/DOCX PARSING FUNCTIONS
# ============================================================

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file."""
    if not PDF_SUPPORT:
        raise ImportError("PyPDF2 not installed")
    
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {str(e)}")
    
    if not text.strip():
        raise ValueError("PDF contains no extractable text")
    
    return text

def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from DOCX file."""
    if not DOCX_SUPPORT:
        raise ImportError("python-docx not installed")
    
    text = ""
    try:
        doc = docx.Document(io.BytesIO(file_content))
        for paragraph in doc.paragraphs:
            if paragraph.text:
                text += paragraph.text + "\n"
    except Exception as e:
        raise ValueError(f"Failed to parse DOCX: {str(e)}")
    
    return text

def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """Extract text from file based on extension."""
    ext = filename.lower().split('.')[-1]
    
    if ext == 'pdf':
        return extract_text_from_pdf(file_content)
    elif ext == 'docx':
        return extract_text_from_docx(file_content)
    elif ext == 'txt':
        return file_content.decode('utf-8', errors='ignore')
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported: pdf, docx, txt")

# ============================================================
# TEXT PROCESSING FUNCTIONS
# ============================================================

def clean_and_lemmatize(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def extract_skills_with_patterns(text, skill_patterns):
    """Extract skills using provided patterns."""
    text_lower = text.lower()
    found = []
    for skill, pattern in skill_patterns.items():
        if re.search(pattern, text_lower):
            found.append(skill)
    return found

def calculate_skill_score_with_weights(job_skills, resume_skills, skill_weights):
    """Calculate weighted skill match score."""
    total_weight = 0
    matched_weight = 0
    for skill in job_skills:
        weight = skill_weights.get(skill, 1)
        total_weight += weight
        if skill in resume_skills:
            matched_weight += weight
    return matched_weight / total_weight if total_weight > 0 else 0

def generate_recommendation(final_score, matched_count, total_required, missing_skills):
    if final_score >= 0.7:
        return f"Strong match ({final_score:.0%}). Candidate meets {matched_count}/{total_required} core requirements. Recommend interview."
    elif final_score >= 0.5:
        return f"Moderate match ({final_score:.0%}). Missing skills: {', '.join(missing_skills[:3])}. Consider further screening."
    else:
        return f"Low match ({final_score:.0%}). Significant skill gaps: {', '.join(missing_skills[:5])}. Not recommended."

def process_resume_text(job_description, resume_text, semantic_weight=0.7, skill_weight=0.3, 
                        skill_patterns=None, skill_weights=None):
    """Process a single resume text and return scores using provided patterns."""
    if skill_patterns is None:
        skill_patterns = DEFAULT_SKILL_PATTERNS
    if skill_weights is None:
        skill_weights = DEFAULT_SKILL_WEIGHTS
    
    job_cleaned = clean_and_lemmatize(job_description)
    job_embedding = model.encode([job_cleaned])
    job_skills = extract_skills_with_patterns(job_cleaned, skill_patterns)
    
    resume_cleaned = clean_and_lemmatize(resume_text)
    resume_embedding = model.encode([resume_cleaned])
    semantic_score = cosine_similarity(job_embedding, resume_embedding)[0][0]
    
    resume_skills = extract_skills_with_patterns(resume_cleaned, skill_patterns)
    skill_score = calculate_skill_score_with_weights(job_skills, resume_skills, skill_weights)
    final_score = (semantic_weight * semantic_score) + (skill_weight * skill_score)
    
    matched = [s for s in job_skills if s in resume_skills]
    missing = [s for s in job_skills if s not in resume_skills]
    recommendation = generate_recommendation(final_score, len(matched), len(job_skills), missing)
    
    return {
        "semantic_score": float(semantic_score),
        "skill_score": float(skill_score),
        "composite_score": float(final_score),
        "matched_skills": matched,
        "missing_skills": missing,
        "recommendation": recommendation
    }

# ============================================================
# PYDANTIC SCHEMAS
# ============================================================

class ResumeInput(BaseModel):
    candidate_id: str
    name: Optional[str] = None
    text: str

class ResumeFileInput(BaseModel):
    candidate_id: str
    name: Optional[str] = None
    filename: str
    content_base64: str

class ScreenRequest(BaseModel):
    job_description: str
    resumes: List[ResumeInput]

class ScreenFileRequest(BaseModel):
    job_description: str
    resumes: List[ResumeFileInput]

class RankedCandidate(BaseModel):
    rank: int
    candidate_id: str
    name: Optional[str]
    candidate_name: Optional[str] = None
    category: Optional[str] = None
    semantic_score: float
    skill_score: float
    composite_score: float
    matched_skills: List[str]
    missing_skills: List[str]
    recommendation: str

class ScreenResponse(BaseModel):
    job_reference: str
    assessment_date: str
    total_candidates: int
    ranked_candidates: List[RankedCandidate]

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(title="Resume Screening API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Resume Screening API", "version": "2.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

# ============================================================
# JSON + Base64 ENDPOINTS
# ============================================================

@app.post("/screen", response_model=ScreenResponse)
def screen_resumes(request: ScreenRequest):
    """Screen resumes with plain text input."""
    results = []
    for resume in request.resumes:
        score_data = process_resume_text(request.job_description, resume.text)
        results.append({
            "candidate_id": resume.candidate_id,
            "name": resume.name,
            "candidate_name": resume.name,
            **score_data
        })
    
    results.sort(key=lambda x: x["composite_score"], reverse=True)
    for i, r in enumerate(results, 1):
        r["rank"] = i
    
    return ScreenResponse(
        job_reference=f"JOB-{str(uuid.uuid4())[:8].upper()}",
        assessment_date=datetime.now().isoformat(),
        total_candidates=len(results),
        ranked_candidates=[RankedCandidate(**r) for r in results]
    )

@app.post("/screen-files", response_model=ScreenResponse)
def screen_resumes_from_files(request: ScreenFileRequest):
    """Screen resumes with file uploads (PDF, DOCX, TXT) via JSON + Base64."""
    results = []
    
    for resume_file in request.resumes:
        try:
            file_content = base64.b64decode(resume_file.content_base64)
            resume_text = extract_text_from_file(file_content, resume_file.filename)
            score_data = process_resume_text(request.job_description, resume_text)
            
            results.append({
                "candidate_id": resume_file.candidate_id,
                "name": resume_file.name,
                "candidate_name": resume_file.name,
                **score_data
            })
        except Exception as e:
            results.append({
                "candidate_id": resume_file.candidate_id,
                "name": resume_file.name,
                "candidate_name": resume_file.name,
                "semantic_score": 0.0,
                "skill_score": 0.0,
                "composite_score": 0.0,
                "matched_skills": [],
                "missing_skills": [],
                "recommendation": f"Error parsing file: {str(e)}"
            })
    
    results.sort(key=lambda x: x["composite_score"], reverse=True)
    for i, r in enumerate(results, 1):
        r["rank"] = i
    
    return ScreenResponse(
        job_reference=f"JOB-{str(uuid.uuid4())[:8].upper()}",
        assessment_date=datetime.now().isoformat(),
        total_candidates=len(results),
        ranked_candidates=[RankedCandidate(**r) for r in results]
    )

# ============================================================
# FORMDATA ENDPOINT (For Frontend)
# ============================================================

@app.post("/screen-form")
async def screen_form(
    file: UploadFile = File(...),
    job_description: str = Form(...),
    semantic_weight: float = Form(0.7),
    skill_weight: float = Form(0.3),
    required_skills: str = Form("[]")
):
    """
    Screen a single resume using FormData.
    Accepts required_skills as JSON string from frontend.
    """
    try:
        # Parse required skills from frontend
        frontend_skills = json.loads(required_skills)
        
        # Build skill patterns and weights from frontend skills
        # Each skill gets a default weight of 2 (medium importance)
        skill_patterns = {}
        skill_weights = {}
        
        for skill in frontend_skills:
            # Convert skill to lowercase for pattern matching
            skill_lower = skill.lower()
            # Create word boundary pattern
            skill_patterns[skill_lower] = r'\b' + re.escape(skill_lower) + r'\b'
            # Default weight for frontend skills is 2
            skill_weights[skill_lower] = 2
        
        # If no skills from frontend, use defaults
        if not skill_patterns:
            skill_patterns = DEFAULT_SKILL_PATTERNS
            skill_weights = DEFAULT_SKILL_WEIGHTS
        
        # Read the raw file bytes
        content = await file.read()
        filename = file.filename
        
        # Extract text based on file type
        if filename.endswith('.pdf'):
            resume_text = extract_text_from_pdf(content)
        elif filename.endswith('.docx'):
            resume_text = extract_text_from_docx(content)
        elif filename.endswith('.txt'):
            resume_text = content.decode('utf-8', errors='ignore')
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}")
        
        # Process the resume with frontend skills
        result = process_resume_text(
            job_description, resume_text, 
            semantic_weight, skill_weight,
            skill_patterns, skill_weights
        )
        
        # Extract candidate name from filename
        candidate_name = filename.replace('.pdf', '').replace('.docx', '').replace('.txt', '')
        
        # Return response
        return {
            "semantic_score": result["semantic_score"],
            "skill_score": result["skill_score"],
            "composite_score": result["composite_score"],
            "matched_skills": result["matched_skills"],
            "missing_skills": result["missing_skills"],
            "recommendation": result["recommendation"],
            "candidate_name": candidate_name,
            "category": "Extracted from resume"
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid required_skills format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# BATCH FORMDATA ENDPOINT (For multiple files)
# ============================================================

@app.post("/screen-form-batch")
async def screen_form_batch(
    files: List[UploadFile] = File(...),
    job_description: str = Form(...),
    semantic_weight: float = Form(0.7),
    skill_weight: float = Form(0.3),
    required_skills: str = Form("[]")
):
    """
    Screen multiple resumes using FormData.
    """
    try:
        frontend_skills = json.loads(required_skills)
        skill_patterns = {}
        skill_weights = {}
        
        for skill in frontend_skills:
            skill_lower = skill.lower()
            skill_patterns[skill_lower] = r'\b' + re.escape(skill_lower) + r'\b'
            skill_weights[skill_lower] = 2
        
        if not skill_patterns:
            skill_patterns = DEFAULT_SKILL_PATTERNS
            skill_weights = DEFAULT_SKILL_WEIGHTS
    except:
        skill_patterns = DEFAULT_SKILL_PATTERNS
        skill_weights = DEFAULT_SKILL_WEIGHTS
    
    results = []
    
    for file in files:
        try:
            content = await file.read()
            filename = file.filename
            
            if filename.endswith('.pdf'):
                resume_text = extract_text_from_pdf(content)
            elif filename.endswith('.docx'):
                resume_text = extract_text_from_docx(content)
            elif filename.endswith('.txt'):
                resume_text = content.decode('utf-8', errors='ignore')
            else:
                continue
            
            result = process_resume_text(
                job_description, resume_text, 
                semantic_weight, skill_weight,
                skill_patterns, skill_weights
            )
            candidate_name = filename.replace('.pdf', '').replace('.docx', '').replace('.txt', '')
            
            results.append({
                "candidate_name": candidate_name,
                "filename": filename,
                **result
            })
            
        except Exception as e:
            candidate_name = file.filename.replace('.pdf', '').replace('.docx', '').replace('.txt', '')
            results.append({
                "candidate_name": candidate_name,
                "filename": file.filename,
                "semantic_score": 0.0,
                "skill_score": 0.0,
                "composite_score": 0.0,
                "matched_skills": [],
                "missing_skills": [],
                "recommendation": f"Error: {str(e)}"
            })
    
    results.sort(key=lambda x: x["composite_score"], reverse=True)
    for i, r in enumerate(results, 1):
        r["rank"] = i
    
    return {
        "total_candidates": len(results),
        "ranked_candidates": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)