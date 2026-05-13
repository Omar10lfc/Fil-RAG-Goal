"""
FilGoalBot Preprocessing Pipeline — v4 (Production)
=====================================================
Input:  data/raw/articles.jsonl
Output: data/processed/chunks.jsonl
        data/processed/stats.json

All 9 FilGoal-specific noise patterns handled:
  1. Social share empty links [](https://twitter/facebook...)
  2. Date/byline: 'الأحد، 15 مارس 2026 - 02:08 كتب : FilGoal'
  3. Scoreboard widget: **1**\\ TeamA\\ **1**\\ TeamB\\ **League**
  4. انتهتHH:MM inline score timestamps
  5. Backslash sequences \\\\ (Firecrawl markdown artifact)
  6. News ticker sidebar: 'N دقيقة |' or 'ساعة |' — CUT everything after
  7. Related articles separator '__' — CUT everything after
  8. Markdown bold/italic/headers/links/images
  9. HTML tags, Getty captions, duplicate adjacent phrases

v4 changes:
  - Skip non-football sections: كرة يد / كرة سلة / كرة طائرة
  - video:1 placeholders removed
  - Embedded tweet pic links pic.twitter.com stripped
  - Hashtags stripped (including escaped \\_)
  - Angle-bracket tweet separator > replaced with space
  - FilGoal domain refs filgoal.com/... stripped
  - Client JS artifact at tail stripped
  - HaytersTV promo paragraph stripped
  - English tweet dates (March 14, 2026) stripped
  - YouTube label + iframe block stripped
  - beIN Sports handles @beINSPORTS stripped
  - Mixed bold-italic _**text**_ stripped
  - Emojis removed
  - Tashkeel/tatweel diacritics removed (normalization)
  - Alef variants أ إ آ → ا (normalization)
"""

import argparse
import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("preprocessing")

RAW_FILE    = Path("data/raw/articles.jsonl")
OUTPUT_DIR  = Path("data/processed")
CHUNKS_FILE = OUTPUT_DIR / "chunks.jsonl"
STATS_FILE  = OUTPUT_DIR / "stats.json"
CHUNK_SIZE, CHUNK_OVERLAP = 300, 60

# Sections to skip entirely — not football content
NON_FOOTBALL_SECTIONS = {
    'كرة يد',
    'كرة سلة',
    'كرة طائرة',
    'رياضات أخرى',
}

_AR_DAYS = r'(الأحد|الإثنين|الثلاثاء|الأربعاء|الخميس|الجمعة|السبت)'
_HARAKAT = re.compile(r'[\u064B-\u065F\u0670\u0640]')
_EMOJI   = re.compile(
    r'[\U0001F300-\U0001F9FF'
    r'\U00002600-\U000027FF'
    r'\U0000FE00-\U0000FE0F'
    r'\U0001FA00-\U0001FA9F]+',
    flags=re.UNICODE,
)

_ENGLISH_MONTHS = (
    r'(January|February|March|April|May|June|'
    r'July|August|September|October|November|December)'
)


def clean_filgoal_body(text: str, title: str = '') -> str:
    if not text:
        return ''

    # ── Phase A: pre-backslash patterns ──────────────────────────────────────

    # 1. Empty social share links
    text = re.sub(r'\[\]\(https?://[^\)]+\)\s*', '', text)

    # 2. Date + byline header
    text = re.sub(
        _AR_DAYS + r'[،,]\s*\d{1,2}\s+\w+\s+\d{4}\s*[-–]\s*\d{2}:\d{2}\s*',
        '', text,
    )
    text = re.sub(r'كتب\s*:\s*FilGoal\s*', '', text)

    # 3. Scoreboard widget  **1**\\ TeamA\\ **1**\\ TeamB\\ **League**
    text = re.sub(
        r'\*\*\d+\*\*\s*\\\\\s*[^\*\n]{2,40}\s*\\\\\s*'
        r'\*\*\d+\*\*\s*\\\\\s*[^\*\n]{2,40}\s*\\\\\s*\*\*[^\*\n]+\*\*',
        '', text,
    )

    # 4. انتهتHH:MM
    text = re.sub(r'انتهت\d{2}:\d{2}\s*', '', text)

    # 5. video:N placeholders
    text = re.sub(r'\bvideo:\d+\b', '', text)

    # 6. Embedded tweet pic links
    text = re.sub(r'https?://pic\.twitter\.com/\S+', '', text)
    text = re.sub(r'pic\.twitter\.com/\S+', '', text)

    # 7. Hashtags (plain and escaped-underscore forms)
    text = re.sub(r'#[\w\u0600-\u06FF]+(?:\\_[\w\u0600-\u06FF]+)*', '', text)

    # 8. Angle-bracket tweet separator lines (">")
    text = re.sub(r'(?m)^>\s*', ' ', text)

    # 9. FilGoal domain references — three variants
    text = re.sub(r'https?://(?:www\.)?filgoal\.com/\S*', '', text)
    text = re.sub(r'(?:www\.)?filgoal\.com/\S*', '', text)
    text = re.sub(r'\bfilgoal\.com\b', '', text)

    # ── Phase B: hard cuts ────────────────────────────────────────────────────

    # 10. All backslashes → space (must happen BEFORE ticker/related cuts)
    text = re.sub(r'\\+', ' ', text)

    # 11. CUT news ticker: 'N دقيقة |' or 'ساعة |'
    ticker = re.search(r'\d+\s+دقيقة\s+\||\bساعة\s+\|', text)
    if ticker:
        text = text[:ticker.start()].strip()

    # 12. CUT at __ related articles separator
    dunder = text.find(' __ ')
    if dunder > 200:
        text = text[:dunder].strip()

    # ── Phase C: post-cut cleanup ─────────────────────────────────────────────

    # 13. Remove /articles/NNNNN paths
    text = re.sub(r'/articles/\d+\S*', '', text)

    # 14. Strip "Client" JS artifact at tail
    text = re.sub(r'\bClient\b.*$', '', text, flags=re.DOTALL)

    # 15. HaytersTV promo paragraph
    text = re.sub(
        r'هايترز\s*تي\s*في.*?(?:يوتيوب|YouTube|اشترك)[^\n]*',
        '', text, flags=re.IGNORECASE | re.DOTALL,
    )

    # 16. YouTube label + iframe block
    text = re.sub(r'\bYouTube\b[^\n]*', '', text, flags=re.IGNORECASE)
    text = re.sub(
        r'\d[\d,\.]*\s*(?:مشترك|subscriber)[^\n]*',
        '', text, flags=re.IGNORECASE,
    )

    # 17. beIN Sports handles and attributions
    text = re.sub(r'@beIN\w*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'beIN\s*Sports?\s*(?:عربي|العربية|Arabic)?', '', text, flags=re.IGNORECASE)

    # 18. English tweet dates "March 14, 2026" / "14 March 2026"
    text = re.sub(
        r'\b\d{1,2}\s+' + _ENGLISH_MONTHS + r'\s+\d{4}\b', '', text,
    )
    text = re.sub(
        _ENGLISH_MONTHS + r'\s+\d{1,2},?\s+\d{4}\b', '', text,
    )

    # 19. Mixed bold-italic  _**text**_  or  **_text_**
    text = re.sub(r'_\*\*([^\*\n]+)\*\*_', r'\1', text)
    text = re.sub(r'\*\*_([^_\n]+)_\*\*',  r'\1', text)

    # 20. Emojis
    text = _EMOJI.sub('', text)

    # ── Phase D: standard markdown → plain text ───────────────────────────────

    text = re.sub(r'\*\*([^\*\n]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^\*\n]+)\*',     r'\1', text)
    text = re.sub(r'#{1,6}\s+',           '',    text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'\[\]\([^\)]+\)',      '',    text)
    text = re.sub(r'!\[[^\]]*\]\([^\)]+\)', '',  text)

    # ── Phase E: HTML + captions ──────────────────────────────────────────────

    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'صورة\s*:\s*\S+', '', text)
    text = re.sub(r'Getty\s*(Images?)?', '', text, flags=re.IGNORECASE)

    # ── Phase F: final normalisation ──────────────────────────────────────────

    # Deduplicate repeated adjacent words
    text = re.sub(r'\b(\S{4,})\s+\1\b', r'\1', text)
    # Collapse whitespace
    text = re.sub(r'[ \t]{2,}', ' ', text).strip()
    # Remove title duplicated at start
    if title and len(title) > 10 and text.startswith(title[:20].strip()):
        text = text[len(title[:20]):].strip()

    return text


def normalize_arabic(text: str) -> str:
    """Normalize for embedding — do NOT use on display text."""
    if not text:
        return ''
    text = _HARAKAT.sub('', text)
    for frm, to in [
        ('[أإآٱ]', 'ا'),
        ('ى',      'ي'),
        ('ة',      'ه'),
        ('ؤ',      'و'),
    ]:
        text = re.sub(frm, to, text)
    return re.sub(r'\s+', ' ', text).strip()


EGYPTIAN_TEAMS = {
    'الأهلي':       'al_ahly',
    'الزمالك':      'zamalek',
    'بيراميدز':     'pyramids',
    'الإسماعيلي':   'ismaily',
    'المصري':       'masry',
    'سيراميكا':     'ceramica',
    'طلائع الجيش':  'tala3a',
    'فاركو':        'farco',
    'حرس الحدود':   'haras',
    'إنبي':         'enppi',
    'المقاولون':    'mokawloon',
    'مودرن':        'modern',
    'البنك الأهلي': 'nbe',
    'غزل المحلة':   'ghazl',
    'سموحة':        'smouha',
    'الجونة':       'el_gouna',
}

LEAGUE_KEYWORDS = {
    'premier_league':   ['الدوري الإنجليزي', 'الدوري الإنجليزي الممتاز'],
    'la_liga':          ['الدوري الإسباني', 'لاليغا'],
    'serie_a':          ['الدوري الإيطالي'],
    'bundesliga':       ['الدوري الألماني'],
    'ligue_1':          ['الدوري الفرنسي'],
    'champions_league': ['دوري أبطال أوروبا'],
    'caf_champions':    ['دوري أبطال إفريقيا', 'الكونفدرالية'],
    'egyptian_league':  ['الدوري المصري', 'دوري المحترفين'],
    'saudi_league':     ['الدوري السعودي', 'روشن'],
}


def detect_teams(text: str) -> list:
    seen, result = set(), []
    for ar, en in EGYPTIAN_TEAMS.items():
        if ar in text and en not in seen:
            seen.add(en)
            result.append(en)
    return result


def detect_league(title: str, section: str, body: str) -> str:
    combined = f"{title} {section} {body[:300]}"
    for lid, kws in LEAGUE_KEYWORDS.items():
        if any(kw in combined for kw in kws):
            return lid
    SECT_MAP = {
        'الكرة المصرية':    'egyptian_league',
        'الدوري المصري':    'egyptian_league',
        'الكرة الإفريقية':  'caf_champions',
        'سعودي في الجول':   'saudi_league',
        'الدوري الإنجليزي': 'premier_league',
        'الكرة الأوروبية':  'champions_league',
    }
    for k, v in SECT_MAP.items():
        if k in section:
            return v
    return 'other'


def chunk_text(text: str) -> list:
    words = text.split()
    if len(words) <= CHUNK_SIZE:
        return [text]
    chunks, start = [], 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunks.append(' '.join(words[start:end]))
        if end == len(words):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


TYPE_AR = {
    'lineup':           'تشكيلة',
    'match_result':     'نتيجة مباراة',
    'press_conference': 'مؤتمر صحفي',
    'training':         'تدريب',
    'transfer':         'ميركاتو',
    'article':          'خبر',
}


def build_chunk_text(art: dict, chunk_body: str, idx: int, total: int) -> str:
    parts = [f"عنوان: {art.get('title_norm') or art.get('title', '')}"]
    t = art.get('article_type', 'article')
    if t != 'article':
        parts.append(f"نوع: {TYPE_AR.get(t, t)}")
    if art.get('section'):
        parts.append(f"قسم: {art['section']}")
    if art.get('league', 'other') != 'other':
        parts.append(f"بطولة: {art['league']}")
    if art.get('teams'):
        parts.append(f"الفرق: {' - '.join(art['teams'][:3])}")
    if art.get('pub_date'):
        parts.append(f"تاريخ: {art['pub_date'][:10]}")
    prefix = ' | '.join(parts)
    suffix = f"[جزء {idx+1}/{total}]\n" if total > 1 else ""
    return f"{prefix}\n\n{suffix}{chunk_body}"


def run_pipeline(raw_file: Path = RAW_FILE):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not Path(raw_file).exists():
        log.error(f"Not found: {raw_file}")
        return

    articles, seen_ids = [], set()
    with open(raw_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                art = json.loads(line)
                aid = art.get('article_id')
                if aid and aid not in seen_ids:
                    seen_ids.add(aid)
                    articles.append(art)
            except json.JSONDecodeError:
                pass

    log.info(f"Loaded {len(articles)} articles")

    type_c:   defaultdict[str, int] = defaultdict(int)
    league_c: defaultdict[str, int] = defaultdict(int)
    total_chunks = skipped = skipped_section = 0
    body_lens = []

    with open(CHUNKS_FILE, 'w', encoding='utf-8') as fout:
        for art in articles:
            # ── Filter: skip non-football sections ───────────────────────────
            section = art.get('section', '').strip()
            if section in NON_FOOTBALL_SECTIONS:
                skipped_section += 1
                log.debug(f"Skipped non-football section '{section}': {art.get('article_id')}")
                continue

            title = art.get('title', '')
            bc = clean_filgoal_body(art.get('body', ''), title)
            if len(bc) < 80:
                skipped += 1
                continue

            body_lens.append(len(bc))
            tn     = normalize_arabic(title)
            bn     = normalize_arabic(bc)
            teams  = detect_teams(title + ' ' + bc)
            league = detect_league(title, section, bc)

            enriched = {
                **art,
                'body_clean': bc,
                'title_norm': tn,
                'teams':      teams,
                'league':     league,
            }
            chunks = chunk_text(bn)
            n = len(chunks)

            for i, cb in enumerate(chunks):
                fout.write(json.dumps({
                    'chunk_id':     f"{art['article_id']}_{i}",
                    'article_id':   art['article_id'],
                    'chunk_index':  i,
                    'total_chunks': n,
                    'text':         build_chunk_text(enriched, cb, i, n),
                    'title':        title,
                    'title_norm':   tn,
                    'body_clean':   bc,
                    'section':      section,
                    'article_type': art.get('article_type', 'article'),
                    'pub_date':     art.get('pub_date', ''),
                    'teams':        teams,
                    'league':       league,
                    'tags':         art.get('tags', []),
                    'source_url':   art.get('source_url', ''),
                    'image':        art.get('image', ''),
                }, ensure_ascii=False) + '\n')
                total_chunks += 1

            type_c[art.get('article_type', 'article')] += 1
            league_c[league] += 1

    processed = len(articles) - skipped - skipped_section
    avg = int(sum(body_lens) / len(body_lens)) if body_lens else 0

    stats = {
        'total_articles':          len(articles),
        'articles_processed':      processed,
        'skipped_short_body':      skipped,
        'skipped_non_football':    skipped_section,
        'total_chunks':            total_chunks,
        'avg_body_length':         avg,
        'min_body_length':         min(body_lens) if body_lens else 0,
        'max_body_length':         max(body_lens) if body_lens else 0,
        'article_types':           dict(type_c),
        'league_coverage':         dict(league_c),
        'processed_at':            datetime.now().isoformat(),
    }
    STATS_FILE.write_text(json.dumps(stats, ensure_ascii=False, indent=2))

    log.info("\n Preprocessing complete!")
    log.info(f"   Articles total      : {len(articles)}")
    log.info(f"   Processed           : {processed}")
    log.info(f"   Skipped (short body): {skipped}")
    log.info(f"   Skipped (non-football sections): {skipped_section}  "
             f"{sorted(NON_FOOTBALL_SECTIONS)}")
    log.info(f"   Chunks              : {total_chunks}")
    log.info(f"   Avg body            : {avg} chars")
    log.info(f"   Types               : {dict(type_c)}")
    log.info(f"   Leagues             : {dict(league_c)}")
    log.info(f"   Output              : {CHUNKS_FILE}")
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FilGoalBot Preprocessing Pipeline v4')
    parser.add_argument('--input', default=str(RAW_FILE), help='Path to raw articles.jsonl')
    args = parser.parse_args()
    run_pipeline(Path(args.input))
