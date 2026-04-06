"""
utils/credibility.py — Domain-based credibility scoring for sources.

Scores range 0.0–1.0. Used by the Source Aggregator node to rank
and annotate the final bibliography.
"""

from urllib.parse import urlparse

# High-trust TLD and domain patterns
HIGH_TRUST_TLDS = {".edu", ".gov", ".ac.uk", ".ac.in"}
HIGH_TRUST_DOMAINS = {
    "scholar.google.com", "pubmed.ncbi.nlm.nih.gov", "arxiv.org",
    "nature.com", "science.org", "ieee.org", "acm.org", "springer.com",
    "wiley.com", "jstor.org", "researchgate.net", "ncbi.nlm.nih.gov",
    "who.int", "un.org", "worldbank.org", "oecd.org", "bbc.com",
    "reuters.com", "apnews.com", "theguardian.com", "nytimes.com",
}
MEDIUM_TRUST_DOMAINS = {
    "wikipedia.org", "medium.com", "towardsdatascience.com",
    "techcrunch.com", "wired.com", "arstechnica.com", "theverge.com",
}
LOW_TRUST_PATTERNS = ["blogspot", "wordpress.com", "quora.com", "reddit.com"]


def score_url(url: str) -> float:
    """Return a credibility score 0.0–1.0 for a given URL."""
    if not url or url == "uploaded_document":
        return 0.85  # uploaded docs are trusted (researcher's own material)

    try:
        parsed = urlparse(url)
        hostname = parsed.netloc.lower().lstrip("www.")
    except Exception:
        return 0.4

    # TLD check
    for tld in HIGH_TRUST_TLDS:
        if hostname.endswith(tld):
            return 0.95

    # Exact domain check
    if hostname in HIGH_TRUST_DOMAINS:
        return 0.90
    if hostname in MEDIUM_TRUST_DOMAINS:
        return 0.65

    # Low-trust pattern check
    for pattern in LOW_TRUST_PATTERNS:
        if pattern in hostname:
            return 0.30

    # Default for unknown domains
    return 0.50


def label_for_score(score: float) -> str:
    if score >= 0.85:
        return "High"
    elif score >= 0.60:
        return "Medium"
    else:
        return "Low"
