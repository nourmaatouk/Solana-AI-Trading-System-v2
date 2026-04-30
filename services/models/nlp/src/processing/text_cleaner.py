import re
from bs4 import BeautifulSoup
from typing import Optional, Dict


def clean_text(text: Optional[str]) -> str:
    """
    Cleans raw text by:
    - handling None safely
    - removing HTML tags
    - removing URLs
    - removing emojis and special Unicode characters
    - lowercasing
    - normalizing and trimming spaces
    - filtering out text shorter than 20 tokens
    """
    if not text:
        return ""
        
    # Remove HTML tags safely using Beautiful Soup
    soup = BeautifulSoup(text, "html.parser")
    # separator=" " ensures words on different lines/tags aren't glued together
    text = soup.get_text(separator=" ")
    
    # Remove URLs (http/https and www.)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove emojis and special Unicode characters
    # Retains alphanumeric characters, spaces, and basic punctuation
    text = re.sub(r'[^\w\s\.\!\?\-\,\'\"]', '', text)
    
    # Lowercase text
    text = text.lower()
    
    # Normalize multiple spaces into single spaces and trim outer spaces
    text = ' '.join(text.split())
    
    # Final check: Token count (using simple space splitting)
    tokens = text.split()
    if len(tokens) < 20:
        return ""
        
    return text


def prepare_article_text(article: Dict) -> str:
    """
    Safely coalesces title, description, and text, then scrubs the payload.
    """
    if not article:
        return ""
        
    title = article.get("title", "") or ""
    description = article.get("description", "") or ""
    text_content = article.get("text", "") or ""
    
    # Coalesce elements 
    combined = f"{title} {description} {text_content}"
    
    return clean_text(combined)
