import requests
from typing import List, Dict
from src.config import get_settings


class GdeltFetcher:
    """
    Fetches latest crypto-related articles from GDELT API.
    Maintains a simple synchronous operation returning basic fields.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.base_url = "https://api.gdeltproject.org/api/v2/doc/doc"

    def fetch(self) -> List[Dict]:
        keywords = self.settings.GDELT_KEYWORDS
        if not keywords:
            return []
            
        # Formulate a query matching config keywords (e.g., "SOL" OR "Solana")
        query_str = " OR ".join([f'"{k}"' for k in keywords])
        
        params = {
            "query": query_str,
            "mode": "artlist",
            "format": "json",
            "timespan": f"{self.settings.GDELT_FETCH_INTERVAL_MINUTES}min",
            "sort": "DateDesc"
        }
        
        headers = {
            "User-Agent": "Solana-Trading-Agent/1.0 (Research Pipeline)"
        }

        try:
            response = requests.get(self.base_url, params=params, headers=headers, timeout=20)
            if response.status_code == 429:
                print("WARNING: GDELT API rate limit exceeded (429 Too Many Requests). Skipping fetch.")
                return []
                
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"GDELT fetch error: {e}")
            return []

        articles = data.get("articles", [])
        results = []
        
        for article in articles:
            results.append({
                "source": "gdelt",
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "published_at": article.get("seendate", ""),
                "text": article.get("title", ""),  # GDELT artlist provides titles; fallback for text
                "tone": article.get("tone", None) 
            })
            
        return results
