from typing import List, Dict


def deduplicate_articles(articles: List[Dict]) -> List[Dict]:
    """Remove duplicate articles by URL, then by title."""
    seen_urls: set = set()
    seen_titles: set = set()
    deduped: List[Dict] = []

    for article in articles:
        url = article.get("url") or ""
        title = article.get("title") or ""

        if url and url in seen_urls:
            continue
        if title and title in seen_titles:
            continue

        if url:
            seen_urls.add(url)
        if title:
            seen_titles.add(title)

        deduped.append(article)

    return deduped
