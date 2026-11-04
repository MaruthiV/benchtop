import json
import re
import time
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

import httpx

from benchtop.env import STATE

CACHE = STATE / "arxiv"
API = "https://export.arxiv.org/api/query"
ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
UA = {"User-Agent": "benchtop/0.1 (research tool; polite single-threaded fetches)"}
_last = [0.0]


def _polite(gap: float = 3.1) -> None:
    # arxiv asks for one request every 3 seconds
    wait = _last[0] + gap - time.time()
    if wait > 0:
        time.sleep(wait)
    _last[0] = time.time()


def _entry(node) -> dict:
    raw_id = node.findtext("a:id", namespaces=ATOM).rsplit("/", 1)[-1]
    return {
        "id": re.sub(r"v\d+$", "", raw_id),
        "version": raw_id,
        "title": " ".join(node.findtext("a:title", namespaces=ATOM).split()),
        "abstract": " ".join(node.findtext("a:summary", namespaces=ATOM).split()),
        "published": node.findtext("a:published", namespaces=ATOM),
        "authors": [a.findtext("a:name", namespaces=ATOM) for a in node.findall("a:author", ATOM)],
        "categories": [c.get("term") for c in node.findall("a:category", ATOM)],
        "primary": node.find("arxiv:primary_category", ATOM).get("term"),
    }


def recent(categories=("cs.LG", "stat.ML"), max_results: int = 200, start: int = 0) -> list[dict]:
    query = " OR ".join(f"cat:{c}" for c in categories)
    params = {
        "search_query": query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "start": start,
        "max_results": max_results,
    }
    _polite()
    r = httpx.get(API, params=params, headers=UA, timeout=60)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    return [_entry(e) for e in root.findall("a:entry", ATOM)]


def by_ids(ids: list[str]) -> list[dict]:
    _polite()
    r = httpx.get(API, params={"id_list": ",".join(ids), "max_results": len(ids)}, headers=UA, timeout=60)
    r.raise_for_status()
    return [_entry(e) for e in ET.fromstring(r.text).findall("a:entry", ATOM)]


class _Text(HTMLParser):
    skip = {"script", "style", "nav", "header", "footer", "button"}
    blocks = {"p", "h1", "h2", "h3", "h4", "h5", "li", "figcaption", "tr", "div", "section", "br"}

    def __init__(self):
        super().__init__()
        self.out, self.depth_skip, self.in_math = [], 0, 0

    def handle_starttag(self, tag, attrs):
        if tag in self.skip:
            self.depth_skip += 1
        elif tag == "math":
            alt = dict(attrs).get("alttext")
            self.out.append(f" ${alt}$ " if alt else " ")
            self.in_math += 1
        elif tag in self.blocks:
            self.out.append("\n")
        if tag in ("h1", "h2", "h3", "h4"):
            self.out.append("## ")

    def handle_endtag(self, tag):
        if tag in self.skip:
            self.depth_skip = max(0, self.depth_skip - 1)
        elif tag == "math":
            self.in_math = max(0, self.in_math - 1)
        elif tag in ("td", "th"):
            self.out.append(" | ")

    def handle_data(self, data):
        if not self.depth_skip and not self.in_math:
            self.out.append(data)

    def text(self) -> str:
        joined = "".join(self.out)
        joined = re.sub(r"[ \t]+", " ", joined)
        return re.sub(r"\n\s*\n+", "\n\n", joined).strip()


def fulltext(arxiv_id: str, max_chars: int = 60_000) -> str | None:
    path = CACHE / f"{arxiv_id}.txt"
    if path.exists():
        return path.read_text() or None
    CACHE.mkdir(parents=True, exist_ok=True)
    _polite(1.5)
    try:
        r = httpx.get(f"https://arxiv.org/html/{arxiv_id}", headers=UA, timeout=60, follow_redirects=True)
    except httpx.HTTPError:
        return None
    if r.status_code != 200 or "ltx_document" not in r.text:
        path.write_text("")
        return None
    parser = _Text()
    parser.feed(r.text)
    text = parser.text()[:max_chars]
    path.write_text(text)
    return text


def lookup(arxiv_id: str) -> dict:
    for listing in CACHE.glob("listing-*.json"):
        for paper in json.loads(listing.read_text()):
            if paper["id"] == arxiv_id:
                return paper
    return by_ids([arxiv_id])[0]


def save_listing(papers: list[dict], name: str) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"listing-{name}.json").write_text(json.dumps(papers, indent=1))
