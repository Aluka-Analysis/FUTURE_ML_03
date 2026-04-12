from fastapi import FastAPI, HTTPException
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
import os

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

# Skill patterns
SKILL_PATTERNS = {
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

SKILL_WEIGHTS = {
    'python': 3, 'machine learning': 3, 'tensorflow': 3, 'pytorch': 3,
    'sql': 2, 'data analysis': 2, 'tableau': 2, 'aws': 2, 'docker': 2, 'kubernetes': 2,
    'excel': 1, 'communication': 1, 'leadership': 1
}

def clean_and_lemmatize(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def extract_skills(text):
    text_lower = text.lower()
    found = []
    for skill, pattern in SKILL_PATTERNS.items():
        if re.search(pattern, text_lower):
            found.append(skill)
    return found

def calculate_skill_score(job_skills, resume_skills):
    total_weight = 0
    matched_weight = 0
    for skill in job_skills:
        weight = SKILL_WEIGHTS.get(skill, 1)
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

# FastAPI app
app = FastAPI(title="Resume Screening API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResumeInput(BaseModel):
    candidate_id: str
    name: Optional[str] = None
    text: str

class ScreenRequest(BaseModel):
    job_description: str
    resumes: List[ResumeInput]

class RankedCandidate(BaseModel):
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
    job_reference: str
    assessment_date: str
    total_candidates: int
    ranked_candidates: List[RankedCandidate]

@app.get("/")
def root():
    return {"message": "Resume Screening API", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/screen", response_model=ScreenResponse)
def screen_resumes(request: ScreenRequest):
    job_cleaned = clean_and_lemmatize(request.job_description)
    job_embedding = model.encode([job_cleaned])
    job_skills = extract_skills(job_cleaned)
    
    results = []
    for resume in request.resumes:
        resume_cleaned = clean_and_lemmatize(resume.text)
        resume_embedding = model.encode([resume_cleaned])
        semantic_score = cosine_similarity(job_embedding, resume_embedding)[0][0]
        
        resume_skills = extract_skills(resume_cleaned)
        skill_score = calculate_skill_score(job_skills, resume_skills)
        final_score = (0.7 * semantic_score) + (0.3 * skill_score)
        
        matched = [s for s in job_skills if s in resume_skills]
        missing = [s for s in job_skills if s not in resume_skills]
        recommendation = generate_recommendation(final_score, len(matched), len(job_skills), missing)
        
        results.append({
            "candidate_id": resume.candidate_id,
            "name": resume.name,
            "final_score": round(final_score, 4),
            "skill_score": round(skill_score, 4),
            "semantic_score": round(semantic_score, 4),
            "matched_skills": matched,
            "missing_skills": missing,
            "recommendation": recommendation
        })
    
    results.sort(key=lambda x: x["final_score"], reverse=True)
    for i, r in enumerate(results, 1):
        r["rank"] = i
    
    return ScreenResponse(
        job_reference=f"JOB-{str(uuid.uuid4())[:8].upper()}",
        assessment_date=datetime.now().isoformat(),
        total_candidates=len(results),
        ranked_candidates=[RankedCandidate(**r) for r in results]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)