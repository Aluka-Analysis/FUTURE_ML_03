"""
Prediction logic for resume screening.
Handles embedding generation, skill extraction, and ranking.
"""

import base64
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from .preprocess import clean_and_lemmatize
from .skills import extract_skills, calculate_skill_score
from .parser import extract_text_from_file


class ResumeScreener:
    """Main screening class for ranking resumes against job descriptions."""
    
    def __init__(self, model_path: str = 'models/sentence_transformer_model'):
        """
        Initialize the screener with Sentence Transformer model.
        
        Args:
            model_path: Path to the saved Sentence Transformer model
        """
        self.model = SentenceTransformer(model_path)
        self.is_loaded = True
    
    def _generate_recommendation(self, final_score: float, skill_score: float, 
                                  matched_skills: list, missing_skills: list) -> str:
        """
        Generate human-readable recommendation based on scores.
        
        Args:
            final_score: Combined hybrid score
            skill_score: Weighted skill match score
            matched_skills: List of skills found in resume
            missing_skills: List of skills required but missing
            
        Returns:
            Recommendation string
        """
        if final_score >= 0.7:
            return f"Strong match ({final_score:.0%}). Candidate meets {len(matched_skills)}/{len(matched_skills) + len(missing_skills)} core requirements. Recommend interview."
        elif final_score >= 0.5:
            return f"Moderate match ({final_score:.0%}). Missing skills: {', '.join(missing_skills[:3])}. Consider further screening."
        else:
            return f"Low match ({final_score:.0%}). Significant skill gaps: {', '.join(missing_skills[:5])}. Not recommended for shortlist."
    
    def score_candidate(self, job_description: str, resume_text: str) -> dict:
        """
        Score a single candidate against a job description.
        
        Args:
            job_description: Text of the job description
            resume_text: Text of the resume
            
        Returns:
            Dictionary with scores, skills, and recommendation
        """
        # Clean and lemmatize texts
        job_cleaned = clean_and_lemmatize(job_description)
        resume_cleaned = clean_and_lemmatize(resume_text)
        
        # Calculate semantic similarity
        job_embedding = self.model.encode([job_cleaned])
        resume_embedding = self.model.encode([resume_cleaned])
        semantic_score = cosine_similarity(job_embedding, resume_embedding)[0][0]
        
        # Extract skills
        job_skills = extract_skills(job_cleaned)
        resume_skills = extract_skills(resume_cleaned)
        
        # Calculate skill score
        skill_score = calculate_skill_score(job_skills, resume_skills)
        
        # Hybrid score (70% semantic, 30% skill)
        hybrid_score = (0.7 * semantic_score) + (0.3 * skill_score)
        
        # Determine matched and missing skills
        matched = [s for s in job_skills if s in resume_skills]
        missing = [s for s in job_skills if s not in resume_skills]
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            hybrid_score, skill_score, matched, missing
        )
        
        return {
            'semantic_score': round(float(semantic_score), 4),
            'skill_score': round(float(skill_score), 4),
            'final_score': round(float(hybrid_score), 4),
            'matched_skills': matched,
            'missing_skills': missing,
            'recommendation': recommendation
        }
    
    def process_resume_file(self, candidate_id: str, name: str, filename: str, content_base64: str) -> dict:
        """
        Process a resume file (PDF, DOCX, TXT) and extract text.
        
        Args:
            candidate_id: Unique identifier
            name: Candidate name
            filename: Original filename
            content_base64: Base64 encoded file content
            
        Returns:
            Dictionary with candidate_id, name, and extracted text
        """
        # Decode base64 to bytes
        file_content = base64.b64decode(content_base64)
        
        # Extract text based on file type
        text = extract_text_from_file(file_content, filename)
        
        return {
            'candidate_id': candidate_id,
            'name': name,
            'text': text
        }
    
    def rank_candidates(self, job_description: str, resumes: list) -> list:
        """
        Rank multiple candidates against a job description.
        
        Args:
            job_description: Text of the job description
            resumes: List of resume dictionaries with 'candidate_id', 'name', 'text'
            
        Returns:
            Ranked list of candidates with scores and ranks
        """
        results = []
        
        for resume in resumes:
            try:
                score_data = self.score_candidate(job_description, resume['text'])
                results.append({
                    'candidate_id': resume['candidate_id'],
                    'name': resume.get('name', 'Unknown'),
                    **score_data
                })
            except Exception as e:
                # If scoring fails, include with error
                results.append({
                    'candidate_id': resume['candidate_id'],
                    'name': resume.get('name', 'Unknown'),
                    'error': str(e),
                    'final_score': 0.0
                })
        
        # Sort by final score (descending)
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Add rank
        for i, candidate in enumerate(results, 1):
            candidate['rank'] = i
        
        return results
    
    def rank_candidates_from_files(self, job_description: str, resume_files: list) -> list:
        """
        Rank candidates from file uploads (PDF, DOCX, TXT).
        
        Args:
            job_description: Text of the job description
            resume_files: List of resume file dicts with 'candidate_id', 'name', 'filename', 'content_base64'
            
        Returns:
            Ranked list of candidates with scores and ranks
        """
        # Process files to extract text
        resumes = []
        for file_data in resume_files:
            try:
                resume = self.process_resume_file(
                    file_data['candidate_id'],
                    file_data.get('name'),
                    file_data['filename'],
                    file_data['content_base64']
                )
                resumes.append(resume)
            except Exception as e:
                # If file parsing fails, include error
                resumes.append({
                    'candidate_id': file_data['candidate_id'],
                    'name': file_data.get('name', 'Unknown'),
                    'text': '',
                    'error': str(e)
                })
        
        # Rank candidates
        return self.rank_candidates(job_description, resumes)