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


def calculate_relevance_score(text: str) -> float:
    """
    Calculates a simple relevance score based on keyword frequency.
    Keywords: Solana, SOL, crypto market, ETF, hack, outage, whale, liquidity, regulation, stablecoin, dex, defi.
    """
    if not text:
        return 0.0
        
    keywords = [
        "solana", "sol", "crypto market", "etf", 
        "hack", "exploit", "outage", "whale", "liquidity", 
        "regulation", "stablecoin", "dex", "defi", "sec",
        "bullish", "bearish", "airdrop", "listing"
    ]
    
    text_lower = text.lower()
    score = 0.0
    
    # Simple count of how many unique keywords appear
    for kw in keywords:
        if kw in text_lower:
            # Solana and SOL get much higher weights to ensure they are always kept
            if kw in ["solana", "sol"]:
                score += 2.0
            else:
                score += 1.0
                
    return score


def classify_event_type(text: str) -> str:
    """
    Maps text to specific crypto event categories.
    Categories: hack/security, outage/network, whale, ETF/regulation, partnership, market_move
    """
    if not text:
        return "general"
        
    text_lower = text.lower()
    
    mapping = {
        "hack/security": ["hack", "exploit", "breach", "vulnerability", "attack"],
        "outage/network": ["outage", "restart", "downtime", "congested", "clogged", "halted"],
        "whale": ["whale", "large transaction", "large move", "transfer", "shrimp", "dolphin"],
        "ETF/regulation": ["etf", "sec", "regulation", "lawsuit", "legal", "court", "binance", "coinbase"],
        "partnership": ["partnership", "partnership", "integrated", "ecosystem", "grant", "collaborated"],
        "market_move": ["crash", "pump", "surge", "plummet", "liquidations", "moon", "dump"]
    }
    
    for category, keywords in mapping.items():
        if any(kw in text_lower for kw in keywords):
            return category
            
    return "general"


def get_event_weight(event_type: str) -> float:
    """
    Returns the impact weight for a specific event category.
    Higher values = more potential to move the market.
    """
    weights = {
        "hack/security": 3.0,
        "outage/network": 3.0, # High impact on Solana specially
        "whale": 2.0,
        "partnership": 2.0,
        "ETF/regulation": 1.0,
        "market_move": 1.0,
        "general": 0.5
    }
    return weights.get(event_type, 0.5)
