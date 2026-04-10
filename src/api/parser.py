"""
Document parser for resumes in various formats.
Supports PDF, DOCX, and plain text.
"""

import io
import os
from typing import Optional

# PDF parsing
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("Warning: PyPDF2 not installed. PDF support disabled.")

# DOCX parsing
try:
    import docx
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False
    print("Warning: python-docx not installed. DOCX support disabled.")


def extract_text_from_pdf(file_content: bytes) -> str:
    """
    Extract text from PDF file.
    
    Args:
        file_content: PDF file as bytes
        
    Returns:
        Extracted text
    """
    if not PDF_SUPPORT:
        raise ImportError("PyPDF2 not installed. Run: pip install PyPDF2")
    
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {str(e)}")
    
    return text


def extract_text_from_docx(file_content: bytes) -> str:
    """
    Extract text from DOCX file.
    
    Args:
        file_content: DOCX file as bytes
        
    Returns:
        Extracted text
    """
    if not DOCX_SUPPORT:
        raise ImportError("python-docx not installed. Run: pip install python-docx")
    
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
    """
    Extract text from file based on extension.
    
    Args:
        file_content: File content as bytes
        filename: Original filename (used to detect format)
        
    Returns:
        Extracted text
    """
    ext = os.path.splitext(filename)[1].lower()
    
    if ext == '.pdf':
        return extract_text_from_pdf(file_content)
    elif ext == '.docx':
        return extract_text_from_docx(file_content)
    elif ext == '.txt':
        return file_content.decode('utf-8', errors='ignore')
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported: .pdf, .docx, .txt")