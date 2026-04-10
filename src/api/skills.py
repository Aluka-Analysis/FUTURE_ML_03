"""
Skill extraction and weighting for resume screening.
"""

import re

# Skill patterns with regex
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
    'leadership': r'\bleadership\b',
    'javascript': r'\bjavascript\b',
    'react': r'\breact\b',
    'java': r'\bjava\b',
}

# Skill weights (3 = critical, 2 = important, 1 = nice-to-have)
SKILL_WEIGHTS = {
    'python': 3,
    'machine learning': 3,
    'tensorflow': 3,
    'pytorch': 3,
    'sql': 2,
    'data analysis': 2,
    'tableau': 2,
    'aws': 2,
    'docker': 2,
    'kubernetes': 2,
    'javascript': 2,
    'react': 2,
    'java': 2,
    'excel': 1,
    'communication': 1,
    'leadership': 1,
}


def extract_skills(text: str) -> list:
    """
    Extract skills from text using regex pattern matching.
    
    Args:
        text: Cleaned resume or job description text
        
    Returns:
        List of skills found in the text
    """
    text_lower = text.lower()
    found = []
    for skill, pattern in SKILL_PATTERNS.items():
        if re.search(pattern, text_lower):
            found.append(skill)
    return found


def calculate_skill_score(job_skills: list, resume_skills: list) -> float:
    """
    Calculate weighted skill match score.
    
    Args:
        job_skills: Skills required for the job
        resume_skills: Skills found in the resume
        
    Returns:
        Weighted score between 0 and 1
    """
    total_weight = 0
    matched_weight = 0
    
    for skill in job_skills:
        weight = SKILL_WEIGHTS.get(skill, 1)
        total_weight += weight
        if skill in resume_skills:
            matched_weight += weight
    
    return matched_weight / total_weight if total_weight > 0 else 0