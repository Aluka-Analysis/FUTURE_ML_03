# Model Card: Resume Screening System

## Model Overview

| Attribute | Value |
|-----------|-------|
| Model Type | Semantic Similarity + Weighted Skill Matching |
| Architecture | Sentence Transformer (all-MiniLM-L6-v2) + Hybrid Scoring |
| Task | Resume screening and candidate ranking against job descriptions |
| Training Date | April 2026 |
| Author | Aluka Precious Oluchukwu |

## Model Description

This is not a traditional classification model. It is a similarity-based ranking system that:

1. Converts job descriptions and resumes to 384-dimensional embeddings
2. Calculates cosine similarity between job and each resume
3. Extracts skills using regex pattern matching
4. Computes weighted skill match score
5. Combines both into a hybrid score (70% semantic + 30% skill)
6. Ranks candidates by hybrid score

### Why This Approach

| Alternative | Why Not Used |
|-------------|--------------|
| TF-IDF + Cosine Similarity | Keyword-based, misses meaning |
| BERT fine-tuning | Requires large labeled dataset |
| Classification | This is ranking, not classification |

## Intended Use

| Use Case | Description |
|----------|-------------|
| Primary | Resume screening for recruiters and HR teams |
| Secondary | Candidate skill gap analysis |
| Out of Scope | Final hiring decisions (human review required) |

### Users

- Recruiters
- HR Managers
- Hiring Teams

## Dataset

| Attribute | Value |
|-----------|-------|
| Source | Kaggle Resume Dataset |
| Size | 2,484 resumes |
| Categories | 25 job roles |
| Format | CSV with resume text and category labels |
| Split | No train/test split (similarity-based, not supervised) |

### Preprocessing

| Step | Description |
|------|-------------|
| Lowercase | Convert all text to lowercase |
| Remove special characters | Keep only letters and spaces |
| Remove URLs, emails, phone numbers | Noise reduction |
| Lemmatization | Convert "developed" → "develop" |
| Remove stopwords | Remove common words (English only) |

## Model Architecture

### Embedding Model

| Parameter | Value |
|-----------|-------|
| Model | sentence-transformers/all-MiniLM-L6-v2 |
| Embedding Dimension | 384 |
| Max Sequence Length | 256 tokens |
| Parameters | 22.7 million |
| Size | 80 MB |

### Skill Extraction

Skills are provided dynamically by the user (frontend). Each skill gets a default weight of 2 (medium importance). The system uses regex word boundaries for exact matching.

### Scoring Formula

semantic_score = cosine_similarity(job_embedding, resume_embedding)

skill_score = sum(weight of matched skills) / sum(weight of required skills)

final_score = (0.7 × semantic_score) + (0.3 × skill_score)

## Performance

### Test Results (Precision@K)

| Job Role | Precision@1 | Precision@3 | Precision@5 | Precision@10 |
|----------|-------------|-------------|-------------|--------------|
| HR Manager | 100% | 100% | 100% | 100% |
| Software Engineer | 100% | 33% | 20% | 20% |
| Machine Learning Engineer | 0% | 33% | 20% | 20% |
| Data Scientist | 0% | 0% | 0% | 0% |

### Sample Output

{
  "rank": 1,
  "candidate_name": "John Doe",
  "semantic_score": 0.74,
  "skill_score": 1.0,
  "composite_score": 0.82,
  "matched_skills": ["python", "machine learning"],
  "missing_skills": [],
  "recommendation": "Strong match (82%). Candidate meets 2/2 core requirements. Recommend interview."
}

## Skill Weights

| Weight | Meaning | Example Skills |
|--------|---------|----------------|
| 3 | Critical | Python, Machine Learning, TensorFlow, PyTorch |
| 2 | Important | SQL, Data Analysis, Tableau, AWS, Docker, Kubernetes |
| 1 | Nice-to-have | Excel, Communication, Leadership |

## Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Semantic similarity favors keyword-dense resumes | May overrank verbose resumes | Hybrid scoring with skill weights |
| PDF parsing fails on scanned images | Some resumes cannot be read | Error handling, user feedback |
| Dataset categories overlap | ENGINEERING resumes score high for Data Science jobs | Documented limitation, not model error |
| No database persistence | Results lost on page refresh | localStorage for demo |
| No authentication | Anyone can access HR dashboard | Acceptable for demo, requires auth for production |

## Ethical Considerations

### Fairness

| Principle | Implementation |
|-----------|----------------|
| No hard rejection | All candidates are ranked, none are filtered out |
| Transparency | Shows matched and missing skills |
| Skill-based only | Ignores demographic attributes |
| Human review | Recommends human review of top candidates |

### Bias Mitigation

| Potential Bias | Mitigation |
|----------------|------------|
| Skill dictionary may favor certain backgrounds | Users can add any skills dynamically |
| Semantic similarity may favor certain writing styles | Balanced with skill matching |
| Dataset may have category imbalances | Documented, not used for filtering |

### Responsible AI Statement

This system is a decision support tool, not a decision maker. It is designed to augment human recruiters, not replace them. Final hiring decisions should always involve human judgment.

## Deployment

| Component | Platform |
|-----------|----------|
| API | Hugging Face Spaces (Docker) |
| Frontend | GitHub Pages |
| Inference | CPU (all-MiniLM-L6-v2) |

### Requirements

fastapi
uvicorn
sentence-transformers
scikit-learn
numpy
pydantic
python-multipart
nltk
PyPDF2
python-docx

## Usage Example

### API Request

curl -X POST https://Aluka-P-resume-screening-api.hf.space/screen-form \
  -F "file=@resume.pdf" \
  -F "job_description=We need a Python developer" \
  -F "semantic_weight=0.7" \
  -F "skill_weight=0.3" \
  -F 'required_skills=["Python","Machine Learning"]'

### API Response

{
  "semantic_score": 0.74,
  "skill_score": 1.0,
  "composite_score": 0.82,
  "matched_skills": ["python"],
  "missing_skills": ["machine learning"],
  "recommendation": "Strong match (82%). Candidate meets 1/2 core requirements. Recommend interview.",
  "candidate_name": "resume"
}

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | April 2026 | Initial release with TF-IDF + cosine similarity |
| 2.0.0 | April 2026 | Upgraded to Sentence Transformers + hybrid scoring |
| 2.1.0 | April 2026 | Added dynamic skill extraction from frontend |
| 2.2.0 | April 2026 | Added PDF/DOCX/TXT support |

## Author

Aluka Precious Oluchukwu
Machine Learning Engineer

LinkedIn: https://www.linkedin.com/in/aluka-precious-b222a2356
GitHub: https://github.com/Aluka-Analysis
Email: alukaprecious4@gmail.com

## References

1. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.
2. Kaggle Resume Dataset: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
3. Hugging Face Sentence Transformers: https://huggingface.co/sentence-transformers