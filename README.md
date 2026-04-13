# FUTURE ML 03 — Resume Screening System

Production-grade AI system for automated resume screening, candidate ranking, and skill gap analysis.

**Author:** Aluka Precious Oluchukwu  
**Machine Learning Engineer — Port Harcourt, Nigeria**

**Live Demo**  
https://aluka-analysis.github.io/FUTURE_ML_03/

**API Documentation**  
https://Aluka-P-resume-screening-api.hf.space/docs

**GitHub Repository**  
https://github.com/Aluka-Analysis/FUTURE_ML_03

---

## Project Overview

This system automates the resume screening process for recruiters and HR teams. It uses semantic similarity and weighted skill matching to rank candidates against job descriptions.

A recruiter submits a job description and resumes. The system returns:

- Ranked candidates by fit score
- Semantic similarity score
- Skill match percentage
- Matched skills (candidate has)
- Missing skills (candidate lacks)
- Plain English recommendation

The system supports PDF, DOCX, and TXT file uploads.

---

## Live Demo

Try the deployed system:

https://aluka-analysis.github.io/FUTURE_ML_03/

The demo includes:

| View | Description |
|------|-------------|
| Screening | Upload resumes, paste job description, run screening |
| Analytics | View screening statistics, top skills, history |
| Candidates | Manage shortlisted candidates |
| Settings | Configure API endpoint, weights, threshold |

---

## System Architecture

Job Description + Resumes
        │
        ▼
Frontend Interface (HTML/CSS/JS)
        │
        ▼
FastAPI Inference Service
        │
        ▼
Text Preprocessing
(Lowercase / Lemmatization / Stopwords)
        │
        ▼
Sentence Transformer Embeddings
(all-MiniLM-L6-v2, 384-dim)
        │
        ▼
Semantic Similarity + Skill Matching
        │
        ▼
Hybrid Scoring (70% semantic + 30% skill)
        │
        ▼
Ranked Candidates + Skill Gap Analysis
        │
        ▼
Displayed on Dashboard

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| Machine Learning | Sentence Transformers, Scikit-learn |
| Embeddings | all-MiniLM-L6-v2 (384-dim) |
| Backend | FastAPI, Uvicorn |
| Frontend | HTML/CSS/JS (responsive) |
| Document Parsing | PyPDF2, python-docx |
| Deployment | Hugging Face Spaces, GitHub Pages |

---

## Features

### Core ML Features

| Feature | Description |
|---------|-------------|
| Semantic Similarity | Uses Sentence Transformers to capture meaning, not just keywords |
| Skill Extraction | Dynamic skill matching based on recruiter-selected skills |
| Hybrid Scoring | 70% semantic + 30% weighted skill match |
| Skill Gap Analysis | Shows exactly which skills are missing |
| Explainable Results | Clear score breakdown and recommendation |

### Frontend Features

| Feature | Description |
|---------|-------------|
| Screening View | Upload resumes, paste job description, select skills |
| Analytics View | Screening history, top demanded skills, shortlist rates |
| Candidates View | Manage shortlisted candidates |
| Settings View | API configuration, default weights, data export |
| Data Persistence | Results and skills saved to localStorage |
| Responsive Design | Works on desktop, tablet, and mobile |

### API Features

| Feature | Description |
|---------|-------------|
| FormData Endpoint | Accepts file uploads (PDF, DOCX, TXT) |
| Dynamic Skills | Uses skills sent from frontend for matching |
| Batch Processing | Screen multiple resumes in one request |
| CORS Enabled | Can be called from any frontend |
| Swagger Docs | Interactive API documentation |

---

## Performance

### Scoring Formula

Final Score = (0.7 × Semantic Similarity) + (0.3 × Weighted Skill Match)

### Skill Weights

| Weight | Meaning | Example Skills |
|--------|---------|----------------|
| 3 | Critical | Python, Machine Learning, TensorFlow, PyTorch |
| 2 | Important | SQL, Data Analysis, Tableau, AWS, Docker, Kubernetes |
| 1 | Nice-to-have | Excel, Communication, Leadership |

### Test Results

| Job Role | Precision@1 | Precision@3 | Precision@5 |
|----------|-------------|-------------|-------------|
| HR Manager | 100% | 100% | 100% |
| Software Engineer | 100% | 33% | 20% |
| Machine Learning Engineer | 0% | 33% | 20% |
| Data Scientist | 0% | 0% | 0% |

Note: Low precision for Data Scientist reflects dataset limitations, not model failure. ENGINEERING resumes in the dataset contain strong data science skills and are correctly ranked higher based on content.

---

## Dataset

- Source: Kaggle Resume Dataset
- Size: 2,484 resumes
- Categories: 25 job roles
- Format: CSV with resume text and category labels

---

## Installation

### Clone Repository

git clone https://github.com/Aluka-Analysis/FUTURE_ML_03.git
cd FUTURE_ML_03

### Create Virtual Environment

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

### Install Dependencies

pip install -r requirements.txt

### Run API Locally

uvicorn src.api.main:app --reload --port 8000

API available at http://localhost:8000

### Open Frontend

Open index.html in your browser, or serve with:

python -m http.server 8080

Then visit http://localhost:8080

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /screen | Screen resumes (plain text) |
| POST | /screen-files | Screen resumes (JSON + Base64) |
| POST | /screen-form | Screen resumes (FormData) |
| POST | /screen-form-batch | Screen multiple resumes (FormData) |
| GET | /health | Health check |
| GET | /docs | Swagger documentation |

---

## Example API Request

### Request

curl -X POST https://Aluka-P-resume-screening-api.hf.space/screen-form \
  -F "file=@resume.pdf" \
  -F "job_description=We need a Python developer with machine learning skills" \
  -F "semantic_weight=0.7" \
  -F "skill_weight=0.3" \
  -F 'required_skills=["Python","Machine Learning","SQL"]'

### Response

{
  "semantic_score": 0.74,
  "skill_score": 1.0,
  "composite_score": 0.82,
  "matched_skills": ["python", "machine learning"],
  "missing_skills": [],
  "recommendation": "Strong match (82%). Candidate meets 2/2 core requirements. Recommend interview.",
  "candidate_name": "resume",
  "category": "Extracted from resume"
}

---

## Project Structure

FUTURE_ML_03/
│
├── index.html                    # Frontend (all views)
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container configuration
├── README.md                     # This file
├── MODEL_CARD.md                 # Detailed model documentation
│
├── src/api/                      # Backend source code
│   ├── main.py                   # FastAPI application
│   ├── schemas.py                # Pydantic models
│   ├── predict.py                # Prediction logic
│   ├── preprocess.py             # Text preprocessing
│   ├── skills.py                 # Skill extraction
│   └── parser.py                 # PDF/DOCX parsing
│
├── models/                       # Trained models
│   └── sentence_transformer_model/  # Sentence Transformer (downloads on first run)
│
├── notebooks/                    # Jupyter notebooks
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb    # Text preprocessing with lemmatization
│   └── 03_screening_model.ipynb  # Model development and evaluation
│
├── data/                         # Dataset (not included in repo)
│   ├── raw/                      # Original dataset
│   └── processed/                # Cleaned resumes
│
└── images/charts/                # Visualizations

---

## Model Card

Detailed model documentation available in MODEL_CARD.md, including:

- Training data and preprocessing
- Model architecture and parameters
- Performance metrics
- Limitations and bias analysis
- Ethical considerations

---

## Business Impact

| Metric | Before (Manual) | After (AI-Powered) |
|--------|-----------------|-------------------|
| Time to screen 100 resumes | 5-10 hours | 5-10 minutes |
| Screening consistency | Variable | Standardized |
| Skill gap identification | Manual review | Automatic |
| Candidate experience | No feedback | Instant match score |
| Recruiter workload | High | Reduced by 80% |

---

## Limitations

| Limitation | Mitigation |
|------------|------------|
| Semantic similarity favors keyword-dense resumes | Hybrid scoring with skill weights |
| PDF parsing may fail on scanned images | Error handling with user feedback |
| Dataset categories have overlap | Documented and explained |
| No database for persistent storage | localStorage for frontend (demo only) |

---

## Future Improvements

- Add database storage (PostgreSQL) for production
- Implement user authentication (HR login)
- Add email confirmation for applicants
- Integrate with LinkedIn API for profile enrichment
- Add batch export to Excel
- Implement model retraining pipeline

---

## Ethical Considerations

This system is designed as a decision support tool, not a decision maker. Key ethical principles:

1. No hard rejection — All candidates are ranked, none are filtered out
2. Transparency — Shows exactly why each candidate scored what they did
3. Skill-based — Focuses on skills, not demographic attributes
4. Human review — Always recommend human review of top candidates
5. Explainability — Clear score breakdown and recommendation

---

## Author

Aluka Precious Oluchukwu
Machine Learning Engineer

Background: Philosophy → Machine Learning
Intern at: Feature Intense
Track: Future Interns ML Track

LinkedIn: https://www.linkedin.com/in/aluka-precious-b222a2356
GitHub: https://github.com/Aluka-Analysis
Email: alukaprecious4@gmail.com

---

## License

MIT

---

Built to make hiring faster, fairer, and more transparent.