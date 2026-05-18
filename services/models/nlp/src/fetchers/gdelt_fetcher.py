import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict

import requests
from src.config import get_settings

_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "gdelt_cache.json"
_CACHE_TTL_HOURS = 4


def _save_cache(articles: List[Dict]) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(_CACHE_PATH, "w") as f:
            json.dump({"fetched_at": datetime.now(timezone.utc).isoformat(), "articles": articles}, f)
    except Exception as e:
        print(f"WARNING: Could not write GDELT cache: {e}")


def _load_cache() -> List[Dict]:
    if not _CACHE_PATH.exists():
        return []
    try:
        with open(_CACHE_PATH) as f:
            data = json.load(f)
        fetched_at = datetime.fromisoformat(data["fetched_at"])
        age = datetime.now(timezone.utc) - fetched_at
        if age > timedelta(hours=_CACHE_TTL_HOURS):
            print(f"WARNING: GDELT cache is {age.total_seconds()/3600:.1f}h old (>{_CACHE_TTL_HOURS}h TTL). Skipping.")
            return []
        articles = data.get("articles", [])
        print(f"INFO: Using cached GDELT data ({len(articles)} articles, {age.total_seconds()/60:.0f}min old).")
        return articles
    except Exception as e:
        print(f"WARNING: Could not read GDELT cache: {e}")
        return []


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

        articles = []
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                print(f"GDELT Fetch Attempt {attempt + 1}/{max_retries}...")
                response = requests.get(self.base_url, params=params, headers=headers, timeout=20)
                
                print(f"DEBUG: Status={response.status_code} URL={getattr(response, 'url', self.base_url)}")
                
                if response.status_code == 429:
                    wait_time = (attempt + 1) * 5
                    print(f"WARNING: GDELT Rate Limited. Cooling down {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                if response.status_code != 200:
                    print(f"WARNING: GDELT returned status {response.status_code}. Retrying...")
                    time.sleep(2)
                    continue

                # Defensive JSON parsing
                try:
                    data = response.json()
                except json.JSONDecodeError:
                    print("ERROR: GDELT response was not valid JSON. Checking for HTML error...")
                    if "<html" in response.text.lower():
                        print("Confirmed: GDELT returned HTML (likely Maintenance or Proxy Error).")
                    time.sleep(3)
                    continue

                articles = data.get("articles", [])

                if articles:
                    print(f"SUCCESS: Fetched {len(articles)} articles from GDELT.")
                    break
                else:
                    print("GDELT returned empty list. Retrying...")
                    time.sleep(2)
                    
            except requests.exceptions.Timeout:
                print(f"TIMEOUT: GDELT took too long on attempt {attempt + 1}")
                time.sleep(2)
            except Exception as e:
                print(f"GDELT internal error on attempt {attempt + 1}: {e}")
                time.sleep(2)

        if not articles:
            print("CRITICAL WARNING: GDELT failed all attempts or returned no data. Trying cache...")
            cached = _load_cache()
            if cached:
                return cached
            print("No usable GDELT cache. Pipeline will rely on RSS fallback only.")
            return []

        results = []
        for article in articles:
            results.append({
                "source": "gdelt",
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "published_at": article.get("seendate", ""),
                "text": "",
                "gdelt_tone": article.get("tone", None),
            })

        _save_cache(results)
        return results
