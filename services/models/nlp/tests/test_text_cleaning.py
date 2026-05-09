import pytest
from src.processing.text_cleaner import clean_text, prepare_article_text


def test_clean_text_none_input():
    """Ensure None input is robustly bypassed mapping to empty string."""
    assert clean_text(None) == ""


def test_clean_text_html_removal():
    """Ensure HTML tags are scrubbed via BeautifulSoup."""
    raw = "<p>This is a paragraph with <b>HTML</b> tags that should be removed. We need more words to exceed the twenty token limit required by the rule easily.</p>"
    cleaned = clean_text(raw)
    assert "<p>" not in cleaned
    assert "<b>" not in cleaned
    assert "html" in cleaned
    assert len(cleaned.split()) >= 20


def test_clean_text_urls_removal():
    """Ensure full http, https, and www formatted links are scrubbed out without trace."""
    raw = "Here is a URL http://example.com and https://test.com/path that we want to ensure get removed. Again extending length to reach twenty tokens threshold easily without problems."
    cleaned = clean_text(raw)
    assert "http://example.com" not in cleaned
    assert "https://test.com/path" not in cleaned
    assert "url" in cleaned


def test_clean_text_special_chars_and_emojis():
    """Ensure non-standard characters like emojis and symbols are safely dropped."""
    raw = "Emojis like 😊 🔥 and weird unicode ♠ should disappear. Here are additionally enough standard words to easily ensure the returned length exceeds the minimum twenty token threshold."
    cleaned = clean_text(raw)
    assert "😊" not in cleaned
    assert "🔥" not in cleaned
    assert "♠" not in cleaned
    assert "emojis like" in cleaned


def test_clean_text_whitespace_normalization():
    """Ensure heavy sporadic whitespaces collapse down into single uniform spacings."""
    raw = "This   text      has    way   too   much     whitespace. Padding words to make sure we hit the twenty token limit for the text cleaner to keep it valid."
    cleaned = clean_text(raw)
    assert "  " not in cleaned
    assert cleaned.startswith("this text has way")


def test_clean_text_short_payload_drops():
    """Validate payloads underneath the configured minimal 20 sequence token threshold are nullified."""
    raw = "This text is simply too short to be considered valid."
    cleaned = clean_text(raw)
    assert cleaned == ""


def test_prepare_article_text():
    """Ensure title, description, and raw text sequentially collapse and successfully pass cleaning routines."""
    article = {
        "title": "Main headline goes here.",
        "description": "<p>A short summary of the event.</p>",
        "text": "The full text of the article provides more details. It includes enough words here so that the combination definitely exceeds twenty separate tokens after cleaning completely."
    }
    
    cleaned = prepare_article_text(article)
    assert "main headline" in cleaned
    assert "summary of the" in cleaned
    assert "full text of the article" in cleaned
    assert "<p>" not in cleaned
