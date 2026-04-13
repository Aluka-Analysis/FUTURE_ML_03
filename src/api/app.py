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
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import base64
import io
import json
import spacy

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

# ============================================================
# PROFESSIONAL SKILL EXTRACTOR (No Hardcoded Lists)
# ============================================================

class SkillExtractor:
    def __init__(self):
        """Initialize skill extractor with NLP capabilities."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("spaCy model loaded")
        except:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
            print("spaCy model downloaded and loaded")
    
    def extract(self, text):
        """Extract skills from job description using multiple NLP methods."""
        skills = set()
        text_lower = text.lower()
        
        # Method 1: Extract phrases after skill indicators
        skill_indicators = [
            'experience in', 'knowledge of', 'proficient in', 'skilled in',
            'expertise in', 'familiar with', 'strong', 'hands-on', 'competent in',
            'ability to', 'background in', 'trained in', 'certified in'
        ]
        
        doc = self.nlp(text_lower)
        
        for indicator in skill_indicators:
            if indicator in text_lower:
                parts = text_lower.split(indicator)
                for part in parts[1:]:
                    # Extract up to next period or comma
                    end_chars = ['.', ',', ';', 'and', 'or']
                    extracted = part
                    for end in end_chars:
                        if end in extracted:
                            extracted = extracted.split(end)[0]
                            break
                    # Split into potential skills
                    potential = re.split(r',|\sand\s', extracted[:100])
                    for skill in potential[:5]:
                        cleaned = skill.strip()
                        if 3 <= len(cleaned) <= 30 and cleaned not in stop_words:
                            skills.add(cleaned)
        
        # Method 2: Extract noun phrases that are likely skills
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip()
            # Filter by length and exclude common words
            if 3 <= len(chunk_text) <= 30 and chunk_text not in stop_words:
                # Check if chunk contains skill-related words
                if any(word in chunk_text for word in ['experience', 'knowledge', 'skill', 'proficient', 'expert']):
                    skills.add(chunk_text)
        
        # Method 3: Extract technical terms using TF-IDF
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=20)
            tfidf_matrix = vectorizer.fit_transform([text_lower])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            top_indices = np.argsort(scores)[-15:][::-1]
            for idx in top_indices:
                term = feature_names[idx]
                if 3 <= len(term) <= 30 and term not in stop_words:
                    skills.add(term)
        except:
            pass
        
        # Method 4: Extract capitalized terms (potential proper skills)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        for cap in capitalized[:10]:
            if 3 <= len(cap) <= 30:
                skills.add(cap.lower())
        
        # Clean up skills
        stopwords_list = {'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'for', 'on', 'with', 
                          'by', 'at', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 
                          'have', 'has', 'had', 'do', 'does', 'did', 'but', 'so', 'if', 'then', 
                          'else', 'when', 'where', 'which', 'while', 'using', 'including', 'such'}
        
        filtered_skills = [s for s in skills if s not in stopwords_list and len(s) > 2]
        
        return list(set(filtered_skills))[:15]  # Return top 15 unique skills

# Initialize skill extractor
skill_extractor = SkillExtractor()

# ============================================================
# PDF/DOCX PARSING FUNCTIONS
# ============================================================

def extract_text_from_pdf(file_content: bytes) -> str:
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
    text_lower = text.lower()
    found = []
    for skill, pattern in skill_patterns.items():
        if re.search(pattern, text_lower):
            found.append(skill)
    return found

def calculate_skill_score_with_weights(job_skills, resume_skills, skill_weights):
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
    if skill_patterns is None:
        skill_patterns = {}
    if skill_weights is None:
        skill_weights = {}
    
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
# FASTAPI APP
# ============================================================

app = FastAPI(title="Resume Screening API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Resume Screening API", "version": "3.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

# ============================================================
# FORMDATA ENDPOINT (With Auto Skill Extraction)
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
    Auto-extracts skills from job description if none provided.
    """
    try:
        # Parse required skills from frontend
        frontend_skills = json.loads(required_skills)
        
        # If no skills provided, auto-extract from job description
        if not frontend_skills:
            frontend_skills = skill_extractor.extract(job_description)
            print(f"Auto-extracted skills: {frontend_skills}")
        
        # Build skill patterns and weights
        skill_patterns = {}
        skill_weights = {}
        
        for skill in frontend_skills:
            skill_lower = skill.lower()
            skill_patterns[skill_lower] = r'\b' + re.escape(skill_lower) + r'\b'
            skill_weights[skill_lower] = 2  # Default weight
        
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
        
        # Process the resume
        result = process_resume_text(
            job_description, resume_text, 
            semantic_weight, skill_weight,
            skill_patterns, skill_weights
        )
        
        # Extract candidate name from filename
        candidate_name = filename.replace('.pdf', '').replace('.docx', '').replace('.txt', '')
        
        return {
            "semantic_score": result["semantic_score"],
            "skill_score": result["skill_score"],
            "composite_score": result["composite_score"],
            "matched_skills": result["matched_skills"],
            "missing_skills": result["missing_skills"],
            "recommendation": result["recommendation"],
            "candidate_name": candidate_name,
            "category": "Extracted from resume",
            "extracted_skills_from_jd": frontend_skills if not json.loads(required_skills) else []
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid required_skills format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# BATCH ENDPOINT
# ============================================================

@app.post("/screen-form-batch")
async def screen_form_batch(
    files: List[UploadFile] = File(...),
    job_description: str = Form(...),
    semantic_weight: float = Form(0.7),
    skill_weight: float = Form(0.3),
    required_skills: str = Form("[]")
):
    try:
        frontend_skills = json.loads(required_skills)
        
        if not frontend_skills:
            frontend_skills = skill_extractor.extract(job_description)
        
        skill_patterns = {}
        skill_weights = {}
        
        for skill in frontend_skills:
            skill_lower = skill.lower()
            skill_patterns[skill_lower] = r'\b' + re.escape(skill_lower) + r'\b'
            skill_weights[skill_lower] = 2
        
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
            "ranked_candidates": results,
            "extracted_skills_from_jd": frontend_skills if not json.loads(required_skills) else []
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)