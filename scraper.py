"""
C-Metric Signal Intelligence — Multi-Source Web Scraper Backend
================================================================
Gathers real company intelligence from every available source.
Falls back automatically: website → subpages → DDG news → DDG general →
Crunchbase → LinkedIn → Wikipedia → tech-press site searches.

Quality check: if total useful text < MIN_USEFUL_CHARS, the pipeline keeps
searching through deeper sources until the threshold is met or all sources
are exhausted.

Run:
    pip install fastapi uvicorn httpx beautifulsoup4 lxml
    python scraper.py
"""

import asyncio
import logging
import os
import re
import time
import urllib.parse
from typing import Optional

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, field_validator

# Playwright is used as a JS-rendering fallback for SPA sites.
# It is installed but browsers may need `playwright install chromium` once.
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("scraper")

# ── app ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="C-Metric Scraper", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend (index.html) directly from this same service, so a
# single Render deploy gives one URL for both the UI and the API — no
# separate static-site deploy needed. Falls back to a plain status message
# if index.html isn't sitting next to this script (e.g. local dev where you
# open index.html directly in the browser instead).
_FRONTEND_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")

@app.get("/", include_in_schema=False)
async def frontend():
    if os.path.exists(_FRONTEND_PATH):
        return FileResponse(_FRONTEND_PATH, media_type="text/html")
    return HTMLResponse(
        "<h3>C-Metric Scraper backend is running.</h3>"
        "<p>index.html wasn't found next to scraper.py in this deploy — "
        "open index.html locally instead, pointing its Backend URL field "
        "at this service's address.</p>"
    )

# ── tuning constants ───────────────────────────────────────────────────────────
TIMEOUT           = 14          # seconds per individual HTTP request
PAGE_MAX_CHARS    = 4000        # chars to keep per scraped page
MAX_SUBPAGES      = 5           # internal subpages to crawl per site
NEWS_RESULTS      = 8           # DDG news headlines to fetch
SNIPPET_RESULTS   = 5           # DDG general snippets
MIN_USEFUL_CHARS  = 1200        # if combined text below this, trigger deep fallback
COMBINED_HARD_CAP = 14000       # hard cap before handing to LLM (~3.5k tokens)

# ── DDG circuit breaker ──────────────────────────────────────────────────────
# html.duckduckgo.com gets blocked/throttled on plenty of networks (ISP-level
# filtering, some corporate/antivirus firewalls, etc). When that happens every
# single request hangs for the full timeout instead of failing fast — with
# ~9 sequential DDG queries per company (jobs + social + tech-press), a dead
# DDG can silently add 3-4 *minutes* per company before the pipeline even
# reaches Groq. This breaker detects a dead DDG after a couple of consecutive
# connection failures and skips the rest for a cooldown period instead of
# hammering a dead endpoint for the whole run.
_DDG_TIMEOUT        = 6     # fail fast — DDG is a bonus source, not critical path
_DDG_FAIL_THRESHOLD = 2     # consecutive connection failures before disabling
_DDG_COOLDOWN       = 300   # seconds to skip DDG entirely once marked dead
_ddg_state = {"available": True, "consecutive_failures": 0, "disabled_until": 0.0}

# DDG's html endpoint soft-blocks (returns HTTP 200/202 with an EMPTY results
# page instead of a real error) when it sees many requests land in a short
# window from the same IP — which is exactly what happens when 4 async
# functions each fire several DDG queries "at once" via asyncio.gather.
# This lock forces a minimum spacing between every DDG request, process-wide,
# so DDG sees a human-like request rate instead of a burst of 10 in <1s.
_ddg_pace_lock: Optional[asyncio.Lock] = None
_ddg_last_request_ts = 0.0
_DDG_MIN_INTERVAL = 1.4   # seconds between consecutive DDG requests, globally

async def _ddg_pace():
    global _ddg_pace_lock, _ddg_last_request_ts
    if _ddg_pace_lock is None:
        _ddg_pace_lock = asyncio.Lock()
    async with _ddg_pace_lock:
        now = time.time()
        wait = _DDG_MIN_INTERVAL - (now - _ddg_last_request_ts)
        if wait > 0:
            await asyncio.sleep(wait)
        _ddg_last_request_ts = time.time()

# Browser-like headers for general scraping
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.google.com/",
    "Cache-Control": "no-cache",
}

# Wikipedia requires a descriptive User-Agent per their bot policy
WIKI_HEADERS = {
    "User-Agent": "CMetricScraper/2.0 (contact: ak@c-metric.com) python-httpx",
    "Accept": "application/json",
}

# Internal subpage paths ranked by signal value
SIGNAL_PATH_PATTERNS = [
    "about", "about-us", "company", "team", "leadership", "management",
    "news", "press", "press-release", "media", "newsroom",
    "blog", "insights", "updates", "announcements", "articles",
    "careers", "jobs", "hiring", "work-with-us", "open-positions",
    "product", "products", "solutions", "platform", "services", "technology",
    "customers", "clients", "case-studies", "success-stories", "partners",
    "investor", "investors", "funding",
]

# ── company name normaliser ────────────────────────────────────────────────────
_SOCIAL_SLUG_RE = re.compile(
    r'^https?://(?:www\.)?'
    r'(?:linkedin\.com/(?:company|in|school)|'
    r'twitter\.com|x\.com|facebook\.com|instagram\.com|'
    r'crunchbase\.com/(?:organization|company)|'
    r'angel\.co/company|wellfound\.com/company|'
    r'github\.com|glassdoor\.com/Overview)'
    r'/([^/?#]+)',
    re.IGNORECASE,
)

def clean_company_name(raw: str) -> str:
    """
    If `raw` is a social/professional-network URL, extract the slug and
    convert it to a readable company name (hyphens/underscores → spaces,
    title-case).  Otherwise return `raw` unchanged.

    Examples:
        http://www.linkedin.com/company/agamonhealth  → "Agamonhealth"
        https://linkedin.com/company/open-ai          → "Open Ai"
        https://twitter.com/ibm                       → "Ibm"
        "Acme Corp"                                   → "Acme Corp"
    """
    raw = raw.strip()
    m = _SOCIAL_SLUG_RE.match(raw)
    if m:
        slug = m.group(1)
        # Convert separators to spaces and title-case
        name = re.sub(r'[-_]+', ' ', slug).strip().title()
        return name
    return raw


# ── models ─────────────────────────────────────────────────────────────────────
class ScrapeRequest(BaseModel):
    company: str
    website: Optional[str] = None

    # coerce empty string → None so sending website="" doesn't cause a 422
    @field_validator("website", mode="before")
    @classmethod
    def empty_str_to_none(cls, v):
        if isinstance(v, str) and v.strip() == "":
            return None
        return v


class SourceResult(BaseModel):
    source:  str          # e.g. "Homepage", "Careers page", "Crunchbase", "Wikipedia"
    url:     str
    title:   str
    text:    str
    chars:   int
    status:  str          # "ok" | "empty" | "error"
    error:   Optional[str] = None


class HiringSignal(BaseModel):
    role:        str
    date_hint:   str   # e.g. "posted 3 days ago", "2024-11", "recent" — raw text from source
    source:      str   # e.g. "LinkedIn Jobs", "Indeed", "Glassdoor"
    is_ai_ml:    bool  # True if role is clearly AI/ML/Data/LLM related


class ScrapeResponse(BaseModel):
    company:         str
    combined_text:   str
    total_chars:     int
    sources:         list[SourceResult]
    news_headlines:  list[str]
    hiring_signals:  list[HiringSignal]   # structured AI/ML job posting signals
    pages_fetched:   int
    quality:         str          # "rich" | "adequate" | "thin" | "failed"
    quality_chars:   int
    tried_sources:   list[str]
    errors:          list[str]


# ── HTTP helper ────────────────────────────────────────────────────────────────
async def fetch(client: httpx.AsyncClient, url: str, label: str = "") -> tuple[int, str]:
    """GET url → (status, text). Never raises — returns (-1, '') on error."""
    try:
        r = await client.get(url, headers=HEADERS, timeout=TIMEOUT,
                             follow_redirects=True)
        log.info(f"  {label or url[:60]} → HTTP {r.status_code}")
        return r.status_code, r.text
    except Exception as e:
        log.warning(f"  {label or url[:60]} → FAIL: {e}")
        return -1, ""


# ── HTML → clean text ──────────────────────────────────────────────────────────
def extract_text(html: str, max_chars: int = PAGE_MAX_CHARS) -> tuple[str, str]:
    """
    Return (title, clean_body_text).
    Handles JS-rendered SPAs (React/Next/Vue) by:
    1. Trying normal main/article content containers first.
    2. If that yields < 120 chars, falling back to the full body text.
    3. If still < 60 chars, harvesting title + meta description/og tags as a
       minimum signal so we never return empty-handed.
    """
    soup = BeautifulSoup(html, "lxml")

    # grab title before stripping anything
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # collect meta description / og:description as a fallback text source
    meta_texts: list[str] = []
    for tag in soup.find_all("meta"):
        name    = (tag.get("name") or tag.get("property") or "").lower()
        content = (tag.get("content") or "").strip()
        if content and any(k in name for k in ("description", "og:description",
                                                "twitter:description", "keywords")):
            meta_texts.append(content)

    # also grab any JSON-LD or schema.org description strings
    for tag in soup.find_all("script", type=re.compile(r"application/ld\+json", re.I)):
        try:
            import json as _json
            data = _json.loads(tag.string or "")
            for field in ("description", "name", "headline", "about"):
                if isinstance(data, dict) and data.get(field):
                    meta_texts.append(str(data[field])[:300])
        except Exception:
            pass

    # strip ALL non-text elements for clean extraction
    # Also strip link/meta/head to prevent binary CSS/font data contaminating get_text()
    soup2 = BeautifulSoup(html, "lxml")
    for tag in soup2(["script", "style", "noscript", "svg", "img", "picture",
                      "nav", "footer", "header", "aside", "form",
                      "iframe", "button", "input", "select", "textarea",
                      "figcaption", "figure", "cookie-banner",
                      "link", "meta", "head"]):
        tag.decompose()

    # Only accept block-level content tags — avoid matching <link> or <meta> elements
    # which can be returned by find(class_=...) and yield no text or binary data.
    _BLOCK_TAGS = {"div", "main", "article", "section", "p", "ul", "ol", "table", "span", "body"}

    def _text_of(el) -> str:
        """Extract text from el only if it is a block-level content element."""
        if el is None:
            return ""
        if el.name not in _BLOCK_TAGS:
            return ""
        t = re.sub(r"\s{2,}", " ", el.get_text(separator=" ", strip=True))
        return t

    def _is_garbage(text: str) -> bool:
        """
        True if text looks like binary/compressed data rather than readable prose.
        Checks two signals:
        1. Non-printable control chars (< 0x20, excluding whitespace)
        2. High ratio of non-ASCII chars that are NOT common Unicode punctuation/emoji
           (binary blobs show up as U+0080–U+00FF replacement chars)
        """
        sample = text[:500]
        if not sample:
            return False
        control  = sum(1 for c in sample if ord(c) < 32 and c not in "\n\r\t ")
        non_ascii_not_emoji = sum(1 for c in sample
                                  if 0x0080 <= ord(c) <= 0x00FF)  # Latin-1 supplement — rare in real content
        bad = control + non_ascii_not_emoji
        return bad > len(sample) * 0.08   # 8% threshold

    # attempt 1: full body text (scripts/styles/links already stripped above)
    # This is always tried first — it captures the most text for JS-heavy SPAs.
    body_text = ""
    if soup2.body:
        body_text = re.sub(r"\s{2,}", " ", soup2.body.get_text(separator=" ", strip=True))
    raw = body_text if not _is_garbage(body_text) else ""

    # attempt 2: if body text is empty/garbage, try semantic containers
    if len(raw) < 120:
        for finder in [
            lambda: soup2.find("main"),
            lambda: soup2.find("article"),
            lambda: soup2.find("section"),
            lambda: soup2.find(id=re.compile(r"\b(content|main|primary|__next|__nuxt|app|root)\b", re.I)),
            lambda: soup2.find(class_=re.compile(r"\b(content|main|primary|wrapper|page)\b", re.I)),
        ]:
            el = finder()
            candidate = _text_of(el)
            if candidate and len(candidate) >= 60 and not _is_garbage(candidate):
                raw = candidate
                break

    # attempt 3: whole document text
    if len(raw) < 60:
        doc_text = re.sub(r"\s{2,}", " ", soup2.get_text(separator=" ", strip=True))
        if not _is_garbage(doc_text):
            raw = doc_text

    # attempt 4: title + meta description when all body extraction fails
    if len(raw) < 60 and (title or meta_texts):
        raw = title + ". " + " | ".join(meta_texts)
        raw = re.sub(r"\s{2,}", " ", raw).strip()
        log.info("  extract_text: using meta-only fallback")

    # always prepend the first unique meta description not already in body
    if meta_texts:
        unique_meta = next((m for m in meta_texts if m not in raw), None)
        if unique_meta and len(raw) < max_chars - 200:
            raw = unique_meta + "  " + raw

    return title, raw[:max_chars]


# ── subpage discovery ──────────────────────────────────────────────────────────
def find_subpage_urls(homepage_html: str, base_url: str) -> list[str]:
    """Return up to MAX_SUBPAGES high-signal internal URLs from homepage."""
    soup  = BeautifulSoup(homepage_html, "lxml")
    base  = urllib.parse.urlparse(base_url)
    found: list[tuple[int, str]] = []
    seen: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        abs_url = urllib.parse.urljoin(base_url, href).rstrip("/")
        parsed  = urllib.parse.urlparse(abs_url)

        if parsed.netloc and parsed.netloc.lstrip("www.") != base.netloc.lstrip("www."):
            continue
        if parsed.scheme not in ("http", "https"):
            continue
        path = parsed.path.lower().strip("/")
        if any(path.endswith(ext) for ext in (".pdf", ".jpg", ".png", ".zip", ".mp4", ".xml")):
            continue
        if abs_url in seen:
            continue

        score = sum(1 for p in SIGNAL_PATH_PATTERNS if p in path)
        if score > 0:
            found.append((score, abs_url))
            seen.add(abs_url)

    found.sort(key=lambda x: -x[0])
    return [u for _, u in found[:MAX_SUBPAGES]]


# ── JS-rendering fallback (Playwright) ────────────────────────────────────────

async def fetch_with_js(url: str, max_chars: int = PAGE_MAX_CHARS) -> tuple[str, str]:
    """
    Render the page with a real headless Chromium browser, wait for JS to finish,
    then return (title, visible_text).  Called only when BS4 yields thin content.
    Returns ("", "") on any error.
    """
    if not PLAYWRIGHT_AVAILABLE:
        return "", ""
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            ctx     = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 800},
                java_script_enabled=True,
            )
            page = await ctx.new_page()

            # Block images, fonts, media — we only need text
            await page.route(
                "**/*.{png,jpg,jpeg,gif,webp,svg,ico,woff,woff2,ttf,otf,eot,mp4,mp3,pdf}",
                lambda route: route.abort()
            )

            await page.goto(url, timeout=20_000, wait_until="domcontentloaded")
            # Extra wait for React/Vue/Next to hydrate
            try:
                await page.wait_for_load_state("networkidle", timeout=8_000)
            except Exception:
                pass   # network-idle may not settle — continue anyway

            title = await page.title()
            # Use innerText — it returns only visible text, no HTML
            text  = await page.evaluate("document.body.innerText")
            text  = re.sub(r"\s{2,}", " ", text).strip()

            await browser.close()
            log.info(f"  [JS] Playwright rendered {len(text)} chars from {url}")
            return title, text[:max_chars]

    except Exception as e:
        log.warning(f"  [JS] Playwright failed for {url}: {e}")
        return "", ""


# ── Source scrapers ────────────────────────────────────────────────────────────

# Minimum chars from BS4 before we trigger the JS fallback
_JS_FALLBACK_THRESHOLD = 500

async def scrape_website(client: httpx.AsyncClient, raw_site: str) -> tuple[list[SourceResult], str]:
    """
    Fetch homepage + high-signal subpages.
    If BeautifulSoup yields thin content (< 500 chars), automatically
    falls back to Playwright headless rendering for JS-heavy SPAs.
    Returns (results, homepage_html_for_subpage_discovery).
    """
    results: list[SourceResult] = []
    homepage_html = ""
    if not raw_site:
        return results, homepage_html

    url = raw_site.strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    log.info(f"[WEBSITE] Fetching homepage: {url}")
    status, html = await fetch(client, url, "Homepage")
    if status == 200 and html:
        homepage_html = html
        title, text = extract_text(html)

        # ── JS fallback: if BS4 gives us thin content, use Playwright ────────
        if len(text) < _JS_FALLBACK_THRESHOLD and PLAYWRIGHT_AVAILABLE:
            log.info(f"  [JS] BS4 only got {len(text)} chars — triggering Playwright fallback")
            js_title, js_text = await fetch_with_js(url)
            if len(js_text) > len(text):
                title, text = js_title or title, js_text
                log.info(f"  [JS] Playwright improved to {len(text)} chars")

        results.append(SourceResult(
            source="Homepage", url=url, title=title, text=text,
            chars=len(text), status="ok" if len(text) > 50 else "empty"
        ))
        # crawl subpages
        subpage_urls = find_subpage_urls(html, url)
        log.info(f"[WEBSITE] Found {len(subpage_urls)} subpages to crawl")
        tasks = [fetch(client, u, f"Subpage {u}") for u in subpage_urls]
        sub_results = await asyncio.gather(*tasks)
        for sub_url, (st, sub_html) in zip(subpage_urls, sub_results):
            if st == 200 and sub_html:
                path  = urllib.parse.urlparse(sub_url).path.lower()
                ptype = next((p for p in ["news","press","blog","careers","jobs","about","team","product","solutions"] if p in path), "page")
                sub_title, sub_text = extract_text(sub_html, max_chars=2800)
                # JS fallback for thin subpages too
                if len(sub_text) < _JS_FALLBACK_THRESHOLD and PLAYWRIGHT_AVAILABLE:
                    _, js_sub = await fetch_with_js(sub_url, max_chars=2800)
                    if len(js_sub) > len(sub_text):
                        sub_text = js_sub
                if len(sub_text) > 80:
                    results.append(SourceResult(
                        source=ptype.title() + " page",
                        url=sub_url, title=sub_title, text=sub_text,
                        chars=len(sub_text), status="ok"
                    ))
    elif status != 200:
        results.append(SourceResult(
            source="Homepage", url=url, title="", text="",
            chars=0, status="error",
            error=f"HTTP {status}" if status != -1 else "Connection failed"
        ))

    return results, homepage_html


async def search_ddg(client: httpx.AsyncClient, query: str,
                     label: str, max_results: int = 6) -> tuple[list[str], list[str]]:
    """
    Hit DuckDuckGo HTML search. Returns (headlines, snippets).
    DDG can return 200, 202, or 301 depending on region/cache — treat any
    2xx response as success.

    Guarded by a circuit breaker: if DDG is unreachable (connection errors,
    not just "no results"), skip it after a couple of failures instead of
    burning the full timeout on every one of the ~9 DDG calls per company.
    """
    headlines, snippets = [], []

    now = time.time()
    if not _ddg_state["available"]:
        if now < _ddg_state["disabled_until"]:
            log.info(f"  DDG:{label} → skipped (DDG unreachable this run, "
                      f"retrying again in {int(_ddg_state['disabled_until'] - now)}s)")
            return headlines, snippets
        log.info("  DDG cooldown elapsed — probing DDG again")
        _ddg_state["available"] = True
        _ddg_state["consecutive_failures"] = 0

    enc = urllib.parse.quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={enc}"
    await _ddg_pace()   # space out requests so DDG doesn't see a burst and soft-block
    try:
        r = await client.get(url, headers=HEADERS, timeout=_DDG_TIMEOUT, follow_redirects=True)
        status, html = r.status_code, r.text
        log.info(f"  DDG:{label} → HTTP {status} ({len(html)} bytes)")
        _ddg_state["consecutive_failures"] = 0
    except Exception as e:
        _ddg_state["consecutive_failures"] += 1
        reason = f"{type(e).__name__}: {e}" if str(e) else f"{type(e).__name__} (no detail — likely blocked/reset by network)"
        log.warning(f"  DDG:{label} → FAIL: {reason}")
        if _ddg_state["consecutive_failures"] >= _DDG_FAIL_THRESHOLD:
            _ddg_state["available"]     = False
            _ddg_state["disabled_until"] = now + _DDG_COOLDOWN
            log.warning(f"  DDG marked UNREACHABLE from this network — "
                         f"skipping remaining DDG calls for {_DDG_COOLDOWN}s to save time")
        return headlines, snippets

    # accept any 2xx — DDG sometimes returns 202 Accepted with full HTML body
    if not (200 <= status < 300) or not html:
        return headlines, snippets

    # DDG sometimes returns HTTP 200/202 but with an anti-bot / JS-loader
    # interstitial instead of real results — same generic page for every
    # query, usually ~13-16KB, no actual `.result` markup inside. Detect
    # this by absence of real result containers rather than a fixed byte
    # size (a genuine "no results" page is small; this shell page is not).
    headlines, snippets = _parse_ddg_html(html)
    if not headlines and not snippets:
        is_wall = bool(re.search(r"unusual traffic|anomaly|are you a robot|captcha", html, re.I)) \
                  or ('id="links"' not in html and 'class="result' not in html and len(html) > 5000)
        if is_wall:
            log.warning(f"  DDG:{label} → anti-bot wall page (no real result markup in "
                         f"{len(html)}-byte response) — trying lite.duckduckgo.com fallback")
            headlines, snippets = await _search_ddg_lite(client, query, label, max_results)
            if headlines or snippets:
                return headlines, snippets
            # both endpoints walled — this is structural, not transient; trip breaker
            _ddg_state["consecutive_failures"] += 1
            if _ddg_state["consecutive_failures"] >= _DDG_FAIL_THRESHOLD:
                _ddg_state["available"]      = False
                _ddg_state["disabled_until"] = now + _DDG_COOLDOWN
                log.warning(f"  DDG marked UNREACHABLE (anti-bot wall on both endpoints) — "
                             f"skipping remaining DDG calls for {_DDG_COOLDOWN}s")
            return headlines, snippets

    return headlines[:max_results], snippets[:max_results]


def _parse_ddg_html(html: str) -> tuple[list[str], list[str]]:
    """Parse the classic html.duckduckgo.com/html/ markup."""
    headlines, snippets = [], []
    soup = BeautifulSoup(html, "lxml")
    for el in soup.select(".result__title, .result__a, .result__extras__url"):
        t = el.get_text(strip=True)
        if t and len(t) > 12 and t not in headlines:
            headlines.append(t)
    for el in soup.select(".result__snippet"):
        s = el.get_text(strip=True)
        if s and len(s) > 20 and s not in snippets:
            snippets.append(s)
    return headlines, snippets


async def _search_ddg_lite(client: httpx.AsyncClient, query: str,
                            label: str, max_results: int) -> tuple[list[str], list[str]]:
    """
    Fallback search using lite.duckduckgo.com/lite/ — a much simpler,
    table-based static page that historically survives anti-bot changes
    better than the html.duckduckgo.com/html/ endpoint.
    """
    headlines, snippets = [], []
    enc = urllib.parse.quote_plus(query)
    url = f"https://lite.duckduckgo.com/lite/?q={enc}"
    await _ddg_pace()
    try:
        r = await client.get(url, headers=HEADERS, timeout=_DDG_TIMEOUT, follow_redirects=True)
        status, html = r.status_code, r.text
        log.info(f"  DDG-lite:{label} → HTTP {status} ({len(html)} bytes)")
    except Exception as e:
        log.warning(f"  DDG-lite:{label} → FAIL: {type(e).__name__}: {e or 'no detail'}")
        return headlines, snippets

    if not (200 <= status < 300) or not html:
        return headlines, snippets

    soup = BeautifulSoup(html, "lxml")
    # lite.duckduckgo.com uses <a class="result-link"> for titles and
    # <td class="result-snippet"> for snippets, inside a plain <table>.
    for el in soup.select("a.result-link"):
        t = el.get_text(strip=True)
        if t and len(t) > 12 and t not in headlines:
            headlines.append(t)
    for el in soup.select("td.result-snippet"):
        s = el.get_text(strip=True)
        if s and len(s) > 20 and s not in snippets:
            snippets.append(s)

    if not headlines and not snippets and len(html) > 800:
        log.info(f"  DDG-lite:{label} → also 0 results from {len(html)}-byte page; "
                  f"sample: {html[:200]!r}")

    return headlines[:max_results], snippets[:max_results]


async def search_ddg_news(client: httpx.AsyncClient, company: str) -> list[str]:
    """Recent news headlines from DuckDuckGo."""
    q = f"{company} news funding announcement product launch 2024 2025"
    headlines, _ = await search_ddg(client, q, "news", NEWS_RESULTS)
    return headlines


async def search_ddg_general(client: httpx.AsyncClient, company: str) -> list[str]:
    """General company snippets — Wikipedia, Crunchbase, LinkedIn blurbs."""
    q = f"{company} company technology software"
    _, snippets = await search_ddg(client, q, "general", SNIPPET_RESULTS)
    return snippets


async def scrape_crunchbase(client: httpx.AsyncClient, company: str) -> Optional[SourceResult]:
    """
    Try to fetch the Crunchbase organization page for the company.
    Crunchbase slug = company name lowercased, spaces → hyphens.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", company.lower().strip()).strip("-")
    url  = f"https://www.crunchbase.com/organization/{slug}"
    log.info(f"[CRUNCHBASE] Trying: {url}")
    status, html = await fetch(client, url, "Crunchbase")
    if status != 200 or not html:
        return None
    title, text = extract_text(html, max_chars=3000)
    # Crunchbase renders via JS — we often get little text; accept if >100 chars
    if len(text) < 100:
        return None
    return SourceResult(
        source="Crunchbase", url=url, title=title,
        text=text, chars=len(text), status="ok"
    )


async def scrape_wikipedia(client: httpx.AsyncClient, company: str) -> Optional[SourceResult]:
    """
    Fetch Wikipedia summary using the correct bot-friendly User-Agent.
    Two-step: try direct page summary, fall back to opensearch to find the
    right page title, then fetch that summary.
    Wikipedia REST API requires a descriptive UA — generic Chrome UA gives 403.
    """
    import json as _json

    async def _fetch_wiki(url: str, label: str) -> tuple[int, str]:
        """Fetch with the Wikipedia-specific headers."""
        try:
            r = await client.get(url, headers=WIKI_HEADERS, timeout=TIMEOUT,
                                 follow_redirects=True)
            log.info(f"  {label} → HTTP {r.status_code}")
            return r.status_code, r.text
        except Exception as e:
            log.warning(f"  {label} → FAIL: {e}")
            return -1, ""

    def _parse_summary(json_text: str, company: str) -> Optional[SourceResult]:
        try:
            data    = _json.loads(json_text)
            extract = data.get("extract", "").strip()
            if extract and len(extract) > 80:
                page_url = (data.get("content_urls") or {}).get("desktop", {}).get("page", "")
                return SourceResult(
                    source="Wikipedia",
                    url=page_url or f"https://en.wikipedia.org/wiki/{urllib.parse.quote(company)}",
                    title=data.get("title", company),
                    text=extract[:3000],
                    chars=min(len(extract), 3000),
                    status="ok"
                )
        except Exception:
            pass
        return None

    # Step 1 — direct summary lookup
    enc = urllib.parse.quote(company.replace(" ", "_"))
    s, h = await _fetch_wiki(
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{enc}",
        "Wikipedia-direct"
    )
    if s == 200 and h:
        result = _parse_summary(h, company)
        if result:
            return result

    # Step 2 — opensearch to find canonical title
    enc2 = urllib.parse.quote_plus(company)
    s2, h2 = await _fetch_wiki(
        f"https://en.wikipedia.org/w/api.php?action=opensearch&search={enc2}&limit=3&format=json",
        "Wikipedia-opensearch"
    )
    if s2 == 200 and h2:
        try:
            data = _json.loads(h2)
            titles = data[1] if len(data) > 1 else []
            for candidate in titles:
                enc3 = urllib.parse.quote(candidate.replace(" ", "_"))
                s3, h3 = await _fetch_wiki(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{enc3}",
                    f"Wikipedia-{candidate[:30]}"
                )
                if s3 == 200 and h3:
                    result = _parse_summary(h3, candidate)
                    if result:
                        return result
        except Exception:
            pass

    log.info("  Wikipedia — no result found")
    return None


async def scrape_linkedin_ddg(client: httpx.AsyncClient, company: str) -> Optional[SourceResult]:
    """
    LinkedIn blocks direct scraping. Use DDG to find LinkedIn snippets instead.
    Search: site:linkedin.com/company <company>
    """
    q = f'site:linkedin.com/company "{company}"'
    headlines, snippets = await search_ddg(client, q, "LinkedIn", 4)
    combined = "\n".join(snippets + headlines)
    if len(combined) < 60:
        return None
    return SourceResult(
        source="LinkedIn (via search)",
        url=f"https://www.linkedin.com/company/{company.lower().replace(' ','-')}",
        title=f"LinkedIn snippets for {company}",
        text=combined[:2000],
        chars=len(combined[:2000]),
        status="ok"
    )


async def scrape_tech_news(client: httpx.AsyncClient, company: str) -> list[SourceResult]:
    """
    Search tech-press sources: TechCrunch, VentureBeat, Forbes, Business Wire.
    Uses DDG site: operator — no API key needed.
    """
    sources = [
        ("TechCrunch",   f'site:techcrunch.com "{company}"'),
        ("VentureBeat",  f'site:venturebeat.com "{company}"'),
        ("BusinessWire", f'site:businesswire.com "{company}"'),
        ("Forbes",       f'site:forbes.com "{company}" funding OR launch OR partnership'),
    ]
    results: list[SourceResult] = []
    for name, query in sources:
        headlines, snippets = await search_ddg(client, query, name, 4)
        combined = "\n".join(headlines + snippets)
        if len(combined) > 60:
            results.append(SourceResult(
                source=name,
                url=f"https://duckduckgo.com/?q={urllib.parse.quote_plus(query)}",
                title=f"{name} coverage of {company}",
                text=combined[:2000],
                chars=len(combined[:2000]),
                status="ok"
            ))
    return results


async def scrape_g2_or_glassdoor(client: httpx.AsyncClient, company: str) -> Optional[SourceResult]:
    """
    G2 / Glassdoor snippets via DDG — reveals product category, employee reviews,
    tech stack mentions, company size.
    """
    q = f'("{company}" site:g2.com OR site:glassdoor.com) company overview technology'
    headlines, snippets = await search_ddg(client, q, "G2/Glassdoor", 5)
    combined = "\n".join(snippets + headlines)
    if len(combined) < 60:
        return None
    return SourceResult(
        source="G2 / Glassdoor (via search)",
        url=f"https://duckduckgo.com/?q={urllib.parse.quote_plus(q)}",
        title=f"G2/Glassdoor info for {company}",
        text=combined[:2000],
        chars=len(combined[:2000]),
        status="ok"
    )


# ── AI/ML job classification ───────────────────────────────────────────────────
#
# TIER 1 — DIRECT: role is explicitly AI/ML/Data Science. These are unambiguous
#   signals that the company is building AI/ML capability.
DIRECT_AI_ML_KEYWORDS = [
    # core ML titles
    "machine learning engineer", "ml engineer", "machine learning scientist",
    "machine learning researcher", "machine learning lead", "machine learning manager",
    "ml platform", "mlops", "ml infrastructure", "ml ops",
    # AI titles
    "ai engineer", "ai scientist", "ai researcher", "ai architect",
    "artificial intelligence engineer", "ai/ml", "ai product",
    "generative ai", "gen ai", "llm engineer", "llm researcher",
    # Data Science titles
    "data scientist", "senior data scientist", "lead data scientist",
    "staff data scientist", "principal data scientist", "data science",
    # NLP / Vision / specific domains
    "nlp engineer", "natural language processing", "computer vision engineer",
    "deep learning", "reinforcement learning", "foundation model",
    "large language model", "llm", "rag engineer",
    # Automation & AI-adjacent
    "automation engineer", "ai automation", "intelligent automation",
    "ai implementation", "ai integration",
]

# TIER 2 — ADJACENT: roles that strongly suggest AI/ML infrastructure investment
#   but aren't exclusively AI. A company hiring these alongside business growth
#   signals is worth flagging for C-Metric.
ADJACENT_AI_ML_KEYWORDS = [
    # Data platform / engineering — often feeds ML pipelines
    "data engineer", "data platform", "data infrastructure",
    "analytics engineer", "data architect", "data pipeline",
    "etl engineer", "data warehouse", "data lake",
    # Analytics & BI that implies model usage
    "analytics manager", "bi engineer", "business intelligence",
    "quantitative analyst", "quant analyst", "decision scientist",
    "applied scientist", "research scientist", "research engineer",
    # ML-adjacent engineering
    "platform engineer", "ml platform", "model deployment",
    "feature engineering", "data modelling", "data modeling",
    # AI product / strategy
    "ai product manager", "ai product lead", "head of ai",
    "director of ai", "vp of ai", "chief data officer", "cdo",
    "chief ai officer", "head of data", "head of machine learning",
    "director of data", "vp data",
    # Cloud / infra that often accompanies ML workloads
    "cloud architect", "devops", "site reliability", "sre",
    "platform architect", "solutions architect",
]

def _classify_role(text: str) -> str:
    """
    Returns: 'direct' | 'adjacent' | 'general'
    - direct   = explicitly an AI/ML/Data Science role → is_ai_ml = True
    - adjacent = data/platform/analytics role that signals ML investment → is_ai_ml = True
                 (flagged separately so the LLM can weigh it appropriately)
    - general  = unrelated to AI/ML
    """
    t = text.lower()
    if any(kw in t for kw in DIRECT_AI_ML_KEYWORDS):
        return "direct"
    if any(kw in t for kw in ADJACENT_AI_ML_KEYWORDS):
        return "adjacent"
    return "general"

def _is_ai_ml_role(text: str) -> bool:
    """True for both direct and adjacent AI/ML roles."""
    return _classify_role(text) in ("direct", "adjacent")

def _extract_date_hint(snippet: str) -> str:
    """Pull a date or recency hint out of a job snippet."""
    # patterns: "3 days ago", "posted 2 weeks ago", "2024-11", "Nov 2024", "1 month ago"
    patterns = [
        r'\d+\s+(?:hour|day|week|month)s?\s+ago',
        r'posted\s+\d+\s+\w+\s+ago',
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}',
        r'\d{4}-\d{2}(?:-\d{2})?',
        r'today|yesterday|just posted|new',
    ]
    for pat in patterns:
        m = re.search(pat, snippet, re.I)
        if m:
            return m.group(0)
    return "date unknown"


async def search_ai_ml_jobs(client: httpx.AsyncClient, company: str) -> tuple[list[HiringSignal], Optional[SourceResult]]:
    """
    Search multiple job platforms for AI/ML/Data roles at this company.
    Returns (list of HiringSignal, optional SourceResult for the combined text block).
    Searches: LinkedIn Jobs, Indeed, Glassdoor Jobs, Wellfound (AngelList).
    """
    signals: list[HiringSignal] = []
    all_text_parts: list[str] = []

    job_queries = [
        ("Jobs: LinkedIn+Indeed+Glassdoor",
         f'"{company}" (site:linkedin.com/jobs OR site:indeed.com OR site:glassdoor.com/job) '
         f'(AI OR "machine learning" OR "data scientist" OR LLM OR "artificial intelligence")'),
        ("Jobs: Wellfound+General hiring",
         f'"{company}" (site:wellfound.com OR hiring) '
         f'(AI OR "machine learning" OR "data scientist" OR LLM OR "generative AI") 2024 OR 2025'),
    ]

    for source_name, query in job_queries:
        headlines, snippets = await search_ddg(client, query, f"Jobs:{source_name}", 6)
        combined = headlines + snippets
        for item in combined:
            if len(item) < 15:
                continue
            date_hint = _extract_date_hint(item)
            tier      = _classify_role(item)
            is_ai     = tier in ("direct", "adjacent")
            # prefix the role text with its tier so the LLM sees it clearly
            tier_label = "[DIRECT AI/ML]" if tier == "direct" else "[ADJACENT/DATA]" if tier == "adjacent" else "[GENERAL]"
            labelled_role = f"{tier_label} {item[:190]}"
            signals.append(HiringSignal(
                role=labelled_role,
                date_hint=date_hint,
                source=source_name,
                is_ai_ml=is_ai,
            ))
            all_text_parts.append(f"{tier_label} [{source_name}] {item}")

    if not all_text_parts:
        return signals, None

    direct_count   = sum(1 for s in signals if "[DIRECT AI/ML]" in s.role)
    adjacent_count = sum(1 for s in signals if "[ADJACENT/DATA]" in s.role)
    ai_count       = direct_count + adjacent_count
    text_block     = "\n".join(all_text_parts[:30])
    summary = (
        f"Found {len(signals)} job signals: {direct_count} DIRECT AI/ML, "
        f"{adjacent_count} ADJACENT/DATA, {len(signals)-ai_count} General.\n\n"
        + text_block
    )

    return signals, SourceResult(
        source="Job Postings (multi-platform)",
        url=f"https://duckduckgo.com/?q={urllib.parse.quote_plus(company + ' jobs AI ML')}",
        title=f"Hiring signals for {company}",
        text=summary[:3000],
        chars=len(summary[:3000]),
        status="ok",
    )


async def search_social_media(client: httpx.AsyncClient, company: str) -> Optional[SourceResult]:
    """
    Search social media and community signals:
    Twitter/X mentions, Reddit discussions, ProductHunt launches, HackerNews.
    All via DDG — no API keys needed.
    """
    social_queries = [
        ("Twitter/Reddit",
         f'"{company}" (site:twitter.com OR site:x.com OR site:reddit.com) '
         f'(AI OR funding OR launch OR product OR software OR technology OR startup) -job -hiring'),
        ("HackerNews/ProductHunt",
         f'"{company}" (site:news.ycombinator.com OR site:producthunt.com)'),
    ]
    parts: list[str] = []
    for source_name, query in social_queries:
        headlines, snippets = await search_ddg(client, query, f"Social:{source_name}", 3)
        for item in (headlines + snippets):
            if len(item) > 15:
                parts.append(f"[{source_name}] {item}")

    if not parts:
        return None
    text = "\n".join(parts[:20])
    return SourceResult(
        source="Social Media (Twitter/Reddit/HN/PH)",
        url=f"https://duckduckgo.com/?q={urllib.parse.quote_plus(company + ' social media')}",
        title=f"Social media signals for {company}",
        text=text[:2500],
        chars=len(text[:2500]),
        status="ok",
    )


# ── quality helper ─────────────────────────────────────────────────────────────
def measure_quality(sources: list[SourceResult]) -> int:
    """Total useful characters gathered across all OK sources."""
    return sum(s.chars for s in sources if s.status == "ok" and s.chars > 50)


# ── main scrape endpoint ──────────────────────────────────────────────────────
@app.post("/scrape", response_model=ScrapeResponse)
async def scrape(req: ScrapeRequest):
    raw_company = req.company.strip()
    raw_site    = (req.website or "").strip()

    # If the "company" column contains a LinkedIn/social URL, extract the real name.
    # Also: if no website was provided but the company field IS a URL, use it as
    # the website so tier-1 scraping still works where possible.
    company = clean_company_name(raw_company)
    if not raw_site and raw_company != company:
        # raw_company was a URL — check if it's a real website (not just a social profile)
        # LinkedIn/Twitter profiles aren't scrapeable, so only use as site if it's
        # a resolvable domain (not linkedin.com / twitter.com / etc.)
        _social_domains = ("linkedin.com", "twitter.com", "x.com", "facebook.com",
                           "instagram.com", "crunchbase.com", "angel.co",
                           "wellfound.com", "glassdoor.com")
        if not any(d in raw_company.lower() for d in _social_domains):
            raw_site = raw_company   # treat raw URL as the website

    all_sources:    list[SourceResult]  = []
    tried_sources:  list[str]           = []
    news_headlines: list[str]           = []
    all_hiring:     list[HiringSignal]  = []
    errors: list[str]                   = []

    log.info(f"=== START SCRAPE: {company!r} (raw={raw_company!r}) site={raw_site!r} ===")

    async with httpx.AsyncClient() as client:

        # ── TIER 1: company website ───────────────────────────────────────────
        tried_sources.append("Company website")
        site_results, _ = await scrape_website(client, raw_site)
        all_sources.extend(site_results)
        errors.extend(s.error for s in site_results if s.error)
        quality = measure_quality(all_sources)
        log.info(f"After website: {quality} useful chars")

        # ── TIER 2 (parallel): DDG news/general + AI/ML jobs + social media ──
        # Always run all four — jobs and social are independent of scrape quality
        tried_sources.extend(["DDG News", "DDG General", "AI/ML Job search", "Social media"])
        (news_headlines, general_snippets,
         (job_signals, job_source),
         social_source) = await asyncio.gather(
            search_ddg_news(client, company),
            search_ddg_general(client, company),
            search_ai_ml_jobs(client, company),
            search_social_media(client, company),
        )

        if general_snippets:
            all_sources.append(SourceResult(
                source="DuckDuckGo — general snippets",
                url=f"https://duckduckgo.com/?q={urllib.parse.quote_plus(company)}",
                title=f"Web search snippets: {company}",
                text="\n".join(f"• {s}" for s in general_snippets),
                chars=sum(len(s) for s in general_snippets),
                status="ok"
            ))
        if job_signals:
            all_hiring.extend(job_signals)
        if job_source:
            all_sources.append(job_source)
        if social_source:
            all_sources.append(social_source)

        quality = measure_quality(all_sources)
        log.info(f"After DDG+jobs+social: {quality} useful chars, {len(all_hiring)} hiring signals ({sum(1 for h in all_hiring if h.is_ai_ml)} AI/ML)")

        # ── TIER 3: Wikipedia ────────────────────────────────────────────────
        tried_sources.append("Wikipedia")
        wiki = await scrape_wikipedia(client, company)
        if wiki:
            all_sources.append(wiki)
            quality = measure_quality(all_sources)
            log.info(f"After Wikipedia: {quality} useful chars")

        # ── TIER 4: if still thin → Crunchbase + LinkedIn ────────────────────
        if quality < MIN_USEFUL_CHARS:
            log.info(f"Quality thin ({quality} < {MIN_USEFUL_CHARS}), going deeper…")
            tried_sources.extend(["Crunchbase", "LinkedIn (via DDG)"])
            cb, li = await asyncio.gather(
                scrape_crunchbase(client, company),
                scrape_linkedin_ddg(client, company),
            )
            if cb: all_sources.append(cb)
            if li: all_sources.append(li)
            quality = measure_quality(all_sources)
            log.info(f"After Crunchbase+LinkedIn: {quality} useful chars")

        # ── TIER 5: if still thin → tech press + G2/Glassdoor ────────────────
        if quality < MIN_USEFUL_CHARS:
            log.info(f"Still thin ({quality}), searching tech press…")
            tried_sources.extend(["TechCrunch", "VentureBeat", "BusinessWire", "Forbes", "G2/Glassdoor"])
            press_results, g2 = await asyncio.gather(
                scrape_tech_news(client, company),
                scrape_g2_or_glassdoor(client, company),
            )
            all_sources.extend(press_results)
            if g2: all_sources.append(g2)
            quality = measure_quality(all_sources)
            log.info(f"After tech press: {quality} useful chars")

        # ── TIER 6: last resort — broad DDG fallback ─────────────────────────
        if quality < MIN_USEFUL_CHARS:
            log.info(f"Last resort broad search for {company!r}…")
            tried_sources.append("DDG broad fallback")
            _, broad_snippets = await search_ddg(
                client,
                f"{company} overview about founded employees technology platform",
                "broad-fallback", 8
            )
            broad_headlines_wide, _ = await search_ddg(
                client,
                f"{company} company",
                "broad-fallback-2", 8
            )
            combined_broad = "\n".join(broad_snippets + broad_headlines_wide)
            if len(combined_broad) > 60:
                all_sources.append(SourceResult(
                    source="DDG broad search (fallback)",
                    url=f"https://duckduckgo.com/?q={urllib.parse.quote_plus(company)}",
                    title=f"Broad web search: {company}",
                    text=combined_broad[:3000],
                    chars=len(combined_broad[:3000]),
                    status="ok"
                ))
            quality = measure_quality(all_sources)
            log.info(f"Final quality: {quality} useful chars")

    # ── assemble combined_text for LLM ────────────────────────────────────────
    sections: list[str] = []

    if news_headlines:
        sections.append(
            "=== RECENT NEWS HEADLINES (DuckDuckGo) ===\n"
            + "\n".join(f"• {h}" for h in news_headlines)
        )

    # Hiring signals section — always prominent
    if all_hiring:
        ai_jobs  = [h for h in all_hiring if h.is_ai_ml]
        all_jobs = all_hiring
        hiring_lines = []
        for h in all_jobs[:25]:
            tag = "[AI/ML]" if h.is_ai_ml else "[General]"
            hiring_lines.append(f"  {tag} {h.role} | {h.date_hint} | via {h.source}")
        sections.append(
            f"=== JOB POSTINGS ({len(all_jobs)} total, {len(ai_jobs)} AI/ML-related) ===\n"
            + "\n".join(hiring_lines)
        )

    for s in all_sources:
        if s.status == "ok" and s.chars > 50:
            sections.append(
                f"=== SOURCE: {s.source.upper()} ===\n"
                f"URL: {s.url}\n"
                f"Title: {s.title}\n\n"
                f"{s.text}"
            )

    combined = "\n\n".join(sections)

    # If truly nothing found — explicit fallback notice so LLM knows
    final_chars = measure_quality(all_sources)
    if not combined.strip() or final_chars < 100:
        combined = (
            f"NO_SCRAPE_FALLBACK: No web content found for '{company}' across all sources "
            f"(tried: {', '.join(tried_sources)}). "
            f"Analyse using LLM training knowledge only. "
            f"ALL claims MUST be [SPECULATIVE]. Do not assert needs."
        )

    # quality label
    if   final_chars >= 4000: quality_label = "rich"
    elif final_chars >= 1200: quality_label = "adequate"
    elif final_chars >= 200:  quality_label = "thin"
    else:                     quality_label = "failed"

    ai_ml_count = sum(1 for h in all_hiring if h.is_ai_ml)
    log.info(
        f"=== DONE: {company!r} | quality={quality_label} | chars={final_chars} | "
        f"sources={len([s for s in all_sources if s.status=='ok'])} | "
        f"hiring={len(all_hiring)} (AI/ML={ai_ml_count}) ==="
    )

    return ScrapeResponse(
        company        = company,
        combined_text  = combined[:COMBINED_HARD_CAP],
        total_chars    = final_chars,
        sources        = all_sources,
        news_headlines = news_headlines,
        hiring_signals = all_hiring,
        pages_fetched  = len([s for s in all_sources if s.status == "ok"]),
        quality        = quality_label,
        quality_chars  = final_chars,
        tried_sources  = tried_sources,
        errors         = errors,
    )


# ── health ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "c-metric-scraper", "version": "2.0"}


# ── run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import uvicorn
    # Render (and most cloud hosts) assign a dynamic port via the $PORT env
    # var and route external traffic to it — a hardcoded port will fail to
    # bind and the deploy will never come up healthy. Falls back to 8765
    # for local dev where $PORT isn't set.
    port = int(os.environ.get("PORT", 8765))
    uvicorn.run("scraper:app", host="0.0.0.0", port=port, reload=False)
