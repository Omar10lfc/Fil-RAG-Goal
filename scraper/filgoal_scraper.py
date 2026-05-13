"""
FilGoal Scraper — requests + BeautifulSoup
==========================================================
Drop-in replacement for filgoal_scraper.py v2.
Produces identical output schema — same fields, same JSONL format.

No API key needed. No credits. Unlimited.

Install:
    pip install requests beautifulsoup4 lxml

Usage:
    python -m scraper.filgoal_scraper              # scrape up to 3000 articles
    python -m scraper.filgoal_scraper --max 500    # scrape 500 articles
    python -m scraper.filgoal_scraper --no-resume  # ignore checkpoint, start fresh
"""

import argparse
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ─── Config ───────────────────────────────────────────────────────────────────

FILGOAL_BASE      = "https://www.filgoal.com"
HOMEPAGE_URL      = f"{FILGOAL_BASE}/?top=true"
ARTICLES_PAGE_URL = f"{FILGOAL_BASE}/articles?page={{}}"

DELAY             = 1.5     # seconds between requests — be respectful
MAX_ARTICLES      = 3000
SCAN_BACKWARDS    = 500     # IDs below min seed to scan downward
BATCH_SIZE        = 50      # checkpoint every N articles
CONSEC_FAIL_LIMIT = 30      # reset after this many consecutive 404s in sequential scan

OUTPUT_DIR  = Path("data/raw")
OUTPUT_FILE = OUTPUT_DIR / "articles.jsonl"
DONE_FILE   = OUTPUT_DIR / ".scraped_ids"

# Realistic browser headers — helps avoid 403s on news sites
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ar,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.filgoal.com/",
}

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("filgoal")

# ─── HTTP Session ─────────────────────────────────────────────────────────────

def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update(HEADERS)
    return s

SESSION = make_session()

# ─── Low-level fetch ──────────────────────────────────────────────────────────

def fetch_html(url: str) -> BeautifulSoup | None:
    """
    Fetch a URL and return a BeautifulSoup object.
    Returns None on 404, connection errors, or rate limits.
    """
    try:
        resp = SESSION.get(url, timeout=20)

        if resp.status_code == 404:
            return None
        if resp.status_code == 429:
            log.warning("Rate limited — sleeping 20s")
            time.sleep(20)
            return None

        resp.raise_for_status()
        return BeautifulSoup(resp.content, "lxml")

    except requests.RequestException as e:
        log.debug(f"Request error for {url}: {e}")
        return None

# ─── Meta helpers ─────────────────────────────────────────────────────────────

def _meta(soup: BeautifulSoup, *props: str) -> str:
    """
    Try multiple og/meta property names and return the first non-empty value.
    Handles both property= and name= attributes.
    """
    for prop in props:
        tag = soup.find("meta", attrs={"property": prop}) \
           or soup.find("meta", attrs={"name": prop})
        if tag:
            val = tag.get("content", "").strip()
            if val:
                return val
    return ""

# ─── Seed discovery ───────────────────────────────────────────────────────────

def _extract_ids_from_soup(soup: BeautifulSoup) -> list[int]:
    """Pull all /articles/NNNNN IDs from any page's HTML."""
    ids = set()
    for tag in soup.find_all("a", href=True):
        m = re.search(r'/articles/(\d{4,})', tag["href"])
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids, reverse=True)

def get_seed_ids_from_homepage() -> list[int]:
    log.info("🏠 Scraping homepage for seed article IDs...")
    soup = fetch_html(HOMEPAGE_URL)
    if not soup:
        return []
    ids = _extract_ids_from_soup(soup)
    if ids:
        log.info(f"   Found {len(ids)} IDs. Range: {min(ids)} – {max(ids)}")
    return ids

def get_ids_from_listing_page(page: int) -> list[int]:
    url  = ARTICLES_PAGE_URL.format(page)
    log.info(f"📰 Scraping listing page {page}...")
    soup = fetch_html(url)
    if not soup:
        return []
    ids = _extract_ids_from_soup(soup)
    log.info(f"   Page {page}: {len(ids)} IDs")
    return ids

# ─── Article type detection ───────────────────────────────────────────────────

def detect_article_type(title: str) -> str:
    if any(kw in title for kw in ["تشكيل", "تشكيلة"]):
        return "lineup"
    if any(kw in title for kw in ["انتهت", "نتيجة", "هدف", "فاز", "تعادل"]):
        return "match_result"
    if any(kw in title for kw in ["مؤتمر", "تصريح", "قال", "يؤكد", "يكشف"]):
        return "press_conference"
    if any(kw in title for kw in ["مران", "تدريب", "محاضرة"]):
        return "training"
    if any(kw in title for kw in ["ميركاتو", "انتقال", "صفقة", "عقد", "رحيل", "يضم", "تعاقد"]):
        return "transfer"
    return "article"

# ─── Article parser ───────────────────────────────────────────────────────────

def parse_article(article_id: int, url: str, soup: BeautifulSoup) -> dict | None:
    """
    Extract all fields from a BeautifulSoup-parsed FilGoal article page.
    Returns None if the page is not a real article (nav page, 404, etc.)
    """

    # ── Guard: must be an article URL (og:url check for redirects) ────────────
    og_url = _meta(soup, "og:url")
    final_url = og_url if og_url else url
    if "/articles/" not in final_url:
        log.debug(f"  Skip {article_id}: not an article URL ({final_url[:60]})")
        return None

    # ── Title ─────────────────────────────────────────────────────────────────
    title = (
        _meta(soup, "og:title", "twitter:title")
        or (soup.find("h1").get_text(strip=True) if soup.find("h1") else "")
    )
    # Drop generic site-name-only titles
    if not title or (len(title) < 10 and "في الجول" in title):
        log.debug(f"  Skip {article_id}: no usable title")
        return None

    # ── Body — extract from article content div ────────────────────────────────
    # FilGoal wraps article text in <div class="article-body"> or similar.
    # We try several selectors and fall back to <p> tags in the main area.
    body_el = (
        soup.find("div", class_=re.compile(r'article[-_]?(body|content|text)', re.I))
        or soup.find("div", attrs={"itemprop": "articleBody"})
        or soup.find("article")
    )

    if body_el:
        # Remove script, style, nav, aside noise from within the article div
        for tag in body_el.find_all(["script", "style", "nav", "aside", "figure"]):
            tag.decompose()
        paragraphs = [p.get_text(separator=" ", strip=True)
                      for p in body_el.find_all(["p", "h2", "h3", "li"])
                      if len(p.get_text(strip=True)) > 10]
        body = " ".join(paragraphs).strip()
    else:
        # Fallback: grab all <p> tags from the page body
        paragraphs = [p.get_text(separator=" ", strip=True)
                      for p in soup.find_all("p")
                      if len(p.get_text(strip=True)) > 20]
        body = " ".join(paragraphs).strip()

    # Remove title repeated at start of body
    if body.startswith(title[:30]):
        body = body[len(title):].lstrip()

    if len(body) < 80:
        log.debug(f"  Skip {article_id}: body too short ({len(body)} chars)")
        return None

    # ── Metadata ──────────────────────────────────────────────────────────────
    pub_date = _meta(
        soup,
        "article:published_time",
        "og:article:published_time",
        "datePublished",
        "publishedTime",
    )

    section = _meta(soup, "article:section")

    # Tags — article:tag may repeat as multiple meta tags
    tag_metas = soup.find_all("meta", attrs={"property": "article:tag"})
    tags = [t.get("content", "").strip() for t in tag_metas if t.get("content")]
    if not tags:
        raw = _meta(soup, "keywords")
        tags = [t.strip() for t in raw.split(",") if t.strip()] if raw else []

    image = _meta(soup, "og:image", "twitter:image")
    if isinstance(image, list):
        image = image[0] if image else ""

    return {
        "article_id":   article_id,
        "title":        title,
        "body":         body,
        "section":      section,
        "tags":         tags,
        "article_type": detect_article_type(title),
        "pub_date":     pub_date,
        "image":        image,
        "source_url":   final_url,
        "language":     "ar",
        "scraped_at":   datetime.utcnow().isoformat(),
    }

# ─── Checkpoint ───────────────────────────────────────────────────────────────

def load_done_ids() -> set[int]:
    if DONE_FILE.exists():
        return {int(x) for x in DONE_FILE.read_text().split() if x.strip()}
    return set()

def save_done_ids(done: set[int]) -> None:
    DONE_FILE.write_text("\n".join(str(x) for x in sorted(done)))

# ─── Core scraper ─────────────────────────────────────────────────────────────

def scrape_one(article_id: int) -> dict | None:
    url  = f"{FILGOAL_BASE}/articles/{article_id}"
    soup = fetch_html(url)
    if not soup:
        return None
    return parse_article(article_id, url, soup)


def run_scraper(
    max_articles: int = MAX_ARTICLES,
    start_id: int | None = None,
    end_id: int | None = None,
    resume: bool = True,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    done_ids      = load_done_ids() if resume else set()
    scraped_count = 0
    failed_count  = 0
    consecutive_failures = 0

    # ── Step 1: Collect seed IDs ──────────────────────────────────────────────
    seed_ids = get_seed_ids_from_homepage()
    time.sleep(DELAY)

    for page in range(1, 4):
        seed_ids = sorted(set(seed_ids + get_ids_from_listing_page(page)), reverse=True)
        time.sleep(DELAY)

    if not seed_ids:
        log.error("❌ No seed IDs found. Check internet connection.")
        return

    max_seed = max(seed_ids)
    min_seed = min(seed_ids)
    log.info(f"Seeds: {len(seed_ids)} IDs | range {min_seed}–{max_seed}")

    # ── Step 2: Build scan list ───────────────────────────────────────────────
    if start_id is None:
        start_id = min_seed - SCAN_BACKWARDS
    if end_id is None:
        end_id = max_seed

    scan_range = list(range(end_id, start_id - 1, -1))
    known   = [i for i in seed_ids if start_id <= i <= end_id]
    unknown = [i for i in scan_range if i not in set(known)]
    all_ids = known + unknown

    log.info(f"\n🎯 Plan: {len(known)} seeds + {len(unknown)} sequential IDs")
    log.info(f"   Range: {start_id} → {end_id} | Target: {max_articles}")
    log.info(f"   Already scraped: {len(done_ids)}\n")

    # ── Step 3: Scrape ────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "a", encoding="utf-8") as fout:
        for i, article_id in enumerate(all_ids):
            if scraped_count >= max_articles:
                log.info(f"✅ Target reached: {max_articles} articles")
                break

            if article_id in done_ids:
                continue

            log.info(f"[{scraped_count}/{max_articles}] ID {article_id}...")

            article = scrape_one(article_id)
            done_ids.add(article_id)

            if article:
                fout.write(json.dumps(article, ensure_ascii=False) + "\n")
                fout.flush()
                scraped_count       += 1
                consecutive_failures = 0
                log.info(f"  ✓ [{article['article_type']}] {article['title'][:70]}")
            else:
                failed_count         += 1
                consecutive_failures += 1
                if article_id in set(known):
                    log.warning(f"  ✗ Known seed {article_id} failed!")
                else:
                    log.debug(f"  ✗ {article_id} — 404/nav")

            if scraped_count > 0 and scraped_count % BATCH_SIZE == 0:
                save_done_ids(done_ids)
                log.info(f"💾 Checkpoint: {scraped_count} saved")

            # Reset on long empty sequential runs (gap in IDs is normal)
            if i >= len(known) and consecutive_failures >= CONSEC_FAIL_LIMIT:
                log.warning(f"  {CONSEC_FAIL_LIMIT} consecutive 404s — continuing past gap")
                consecutive_failures = 0

            time.sleep(DELAY)

    save_done_ids(done_ids)
    log.info(f"\n✅ Done! Scraped: {scraped_count} | Failed/skipped: {failed_count}")
    log.info(f"   Output: {OUTPUT_FILE}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FilGoal Scraper (requests + BeautifulSoup)")
    parser.add_argument("--max",       type=int,  default=MAX_ARTICLES)
    parser.add_argument("--start-id",  type=int,  default=None)
    parser.add_argument("--end-id",    type=int,  default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    run_scraper(
        max_articles=args.max,
        start_id=args.start_id,
        end_id=args.end_id,
        resume=not args.no_resume,
    )
