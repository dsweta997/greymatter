"""
Phase 1 ingestion: OpenStax + MedQuAD (+ optional NHS) -> one JSON per doc in corpus/raw/.

Security posture (see SAFETY.md):
- pypdf >= 6.13.0 only; PDF parsing is DoS-hardened but still a trusted first-party file.
- defusedxml for any XML (XXE / billion-laughs safe).
- MedQuAD pulled from the license-scrubbed `lavita/MedQuAD` at a PINNED revision,
  never with trust_remote_code=True.
- Every doc carries its license + attribution in metadata.

Run:  python data_pipeline/01_ingest_corpus.py
Each source is resumable: docs are keyed by md5(text); existing ids are skipped.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

CONFIG_PATH = Path("configs/config.yaml")


# --------------------------------------------------------------------------- #
# Small, individually testable cleaning helpers (Phase 1.2)
# --------------------------------------------------------------------------- #
def strip_html(text: str, pattern: str) -> str:
    text = re.sub(pattern, " ", text)
    # minimal entity handling; keep it simple to avoid ReDoS
    for ent, ch in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&nbsp;", " ")):
        text = text.replace(ent, ch)
    return text


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def doc_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# Every doc has these identical top-level keys; anything source-specific
# (chapter, part, pmcid, ...) goes in the `metadata` bag so the schema never drifts.
_CORE_KEYS = ("title", "text", "source", "license", "attribution")


def write_doc(raw_dir: Path, rec: dict, min_chars: int) -> bool:
    """Write one doc JSON in the uniform schema. Returns True if written."""
    text = normalize_ws(rec["text"])
    if len(text) < min_chars:
        return False
    doc = {
        "id": doc_id(text),
        "title": rec.get("title", ""),
        "text": text,
        "source": rec["source"],
        "license": rec["license"],
        "attribution": rec["attribution"],
        # source-specific fields collected here; core keys excluded
        "metadata": {k: v for k, v in rec.items()
                     if k not in _CORE_KEYS and k not in ("id", "metadata")},
    }
    out = raw_dir / f"{doc['id']}.json"
    if out.exists():
        return False  # resumable: already ingested
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


# --------------------------------------------------------------------------- #
# Source: OpenStax — a LIST of local first-party PDFs (one loop per book)
# --------------------------------------------------------------------------- #
# OpenStax section headings look like "22.3 The Process of Breathing".
_SECTION_RE = re.compile(r"^\s*(\d{1,2}\.\d{1,2})\s+(.+\S)\s*$")
_CHAPTER_RE = re.compile(r"^\s*Chapter\s+\d+\s+(.+\S)\s*$")
# Back-matter / non-teaching outline entries to skip even if leaf nodes.
_SKIP_TITLES = {
    "key terms", "chapter review", "review questions", "critical thinking questions",
    "interactive link questions", "chapter objectives", "references", "index",
    "glossary", "contents", "preface",
}


def _openstax_outline_sections(reader):
    """Walk the PDF outline; yield (chapter, section_title, start_page, end_page).

    Uses the PDF's own table of contents so each doc is a real section with a real
    title. end_page is bounded by the NEXT outline entry of any kind, so a section
    never swallows the chapter's back-matter (Key Terms, Review Questions, etc.).
    """
    leaves = []  # (title, page) in document order — ALL entries, unfiltered

    def walk(items):
        for it in items:
            if isinstance(it, list):
                walk(it)
            else:
                try:
                    pg = reader.get_destination_page_number(it)
                except Exception:
                    pg = None
                if pg is not None:
                    leaves.append((str(it.title).strip(), pg))

    try:
        walk(reader.outline)
    except Exception:
        return []

    current_chapter = ""
    out = []
    for i, (title, pg) in enumerate(leaves):
        ch = _CHAPTER_RE.match(title)
        if ch:
            current_chapter = ch.group(1)
        m = _SECTION_RE.match(title)
        if m and title.strip().lower() not in _SKIP_TITLES:
            end = leaves[i + 1][1] if i + 1 < len(leaves) else pg + 1
            out.append((current_chapter, f"{m.group(1)} {m.group(2)}", pg, max(end, pg + 1)))
    return out


# Pack sentences into ~target_chars chunks (~2-3 paragraphs) without splitting a
# sentence, with a 1-sentence overlap so ideas straddling a boundary aren't lost.
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _chunk_text(text: str, target_chars: int = 1000, overlap_sentences: int = 1):
    sents = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
    chunks, cur, size = [], [], 0
    for s in sents:
        cur.append(s)
        size += len(s) + 1
        if size >= target_chars:
            chunks.append(" ".join(cur))
            cur = cur[-overlap_sentences:] if overlap_sentences else []
            size = sum(len(x) + 1 for x in cur)
    if cur and (not chunks or " ".join(cur) != chunks[-1]):
        chunks.append(" ".join(cur))
    return chunks


# Recurring OpenStax page furniture (running footer/URL) that PDF extraction inlines.
_OPENSTAX_BOILER = re.compile(r"Access for free at openstax\.org", re.IGNORECASE)


def _trim_leading_bleed(body: str, sec_title: str) -> str:
    """Cut any tail of the previous section that precedes this section's heading."""
    pat = re.compile(re.escape(sec_title[:40]).replace(r"\ ", r"\s+"))
    m = pat.search(body[:4000])
    return body[m.start():] if m else body


def _ingest_one_openstax(book: dict, cfg: dict, raw_dir: Path, min_chars: int) -> int:
    if not book.get("enabled"):
        return 0
    pdf_path = Path(book["local_pdf"])
    if not pdf_path.exists():
        print(f"[{book['name']}] PDF not found at {pdf_path}. "
              f"Download the official PDF from openstax.org and drop it there. Skipping.")
        return 0

    from pypdf import PdfReader  # pinned >=6.13.0 in requirements

    reader = PdfReader(str(pdf_path))
    sections = _openstax_outline_sections(reader)
    if not sections:
        print(f"[{book['name']}] no usable outline; skipping (would need page-mode fallback).")
        return 0

    n_pages = len(reader.pages)
    page_text = [None] * n_pages
    strip_re = cfg["security"]["strip_html_regex"]
    target = int(cfg["ingest"].get("openstax_chunk_chars", 1000))

    def get_page(i):
        if 0 <= i < n_pages:
            if page_text[i] is None:
                page_text[i] = reader.pages[i].extract_text() or ""
            return page_text[i]
        return ""

    written = 0
    for chapter, sec_title, start, end in tqdm(sections, desc=f"[{book['name']}] sections"):
        body = " ".join(get_page(p) for p in range(start, end))
        body = strip_html(body, strip_re)
        body = _OPENSTAX_BOILER.sub(" ", body).replace("�", " ")  # footers + bullet glyphs
        body = normalize_ws(_trim_leading_bleed(body, sec_title))
        parts = _chunk_text(body, target_chars=target)
        for pi, chunk in enumerate(parts, 1):
            rec = {
                "title": sec_title,
                "chapter": chapter,
                "part": pi,
                "n_parts": len(parts),
                "text": chunk,
                "source": book["name"],
                "license": book["license"],
                "attribution": book["attribution"],
            }
            written += int(write_doc(raw_dir, rec, min_chars))
    print(f"[{book['name']}] wrote {written} chunk docs")
    return written


def ingest_openstax(cfg: dict, raw_dir: Path, min_chars: int) -> int:
    total = 0
    for book in cfg["sources"].get("openstax_books", []):
        total += _ingest_one_openstax(book, cfg, raw_dir, min_chars)
    return total


# --------------------------------------------------------------------------- #
# Source: MedQuAD (pinned revision, educational-register filter)
# --------------------------------------------------------------------------- #
def _passes_register(question: str, keep, drop) -> bool:
    q = question.lower().strip()
    if any(k in q for k in drop):
        return False
    return any(q.startswith(p) for p in keep)


def ingest_medquad(cfg: dict, raw_dir: Path, min_chars: int) -> int:
    src = cfg["sources"]["medquad"]
    if not src.get("enabled"):
        return 0
    if not src.get("revision"):
        print("[medquad] REFUSING to run without a pinned `revision` in config "
              "(tamper-evidence + reproducibility). Set it and rerun.")
        return 0

    from datasets import load_dataset

    ds = load_dataset(
        src["hf_repo"],
        split="train",
        revision=src["revision"],       # pinned commit hash
        trust_remote_code=False,        # HARD RULE
    )

    keep = [p.lower() for p in src["keep_question_prefixes"]]
    drop = [d.lower() for d in src["drop_question_keywords"]]
    written = 0
    for row in tqdm(ds, desc="[medquad] rows"):
        question = (row.get("question") or "").strip()
        answer = (row.get("answer") or "").strip()
        if not question or not answer:
            continue
        if not _passes_register(question, keep, drop):
            continue
        rec = {
            "title": question[:120],
            "text": f"Question: {question}\n\nAnswer: {answer}",
            "source": src["name"],
            "license": src["license"],
            "attribution": src["attribution"],
        }
        written += int(write_doc(raw_dir, rec, min_chars))
    print(f"[medquad] wrote {written} docs")
    return written


# --------------------------------------------------------------------------- #
# License gate — shared by any source with per-document license variation
# --------------------------------------------------------------------------- #
def license_slug(text: str) -> str | None:
    """Map a license URL/string to a normalized CC slug, or None if unrecognized."""
    t = (text or "").lower()
    if "publicdomain" in t or "/zero/" in t or "cc0" in t:
        return "cc0"
    m = re.search(r"creativecommons\.org/licenses/([a-z-]+)", t)
    if m:
        return m.group(1).strip("-")
    # bare textual forms like "CC BY-NC-SA"
    m = re.search(r"\bcc[ -]?by(-nc)?(-sa|-nd)?\b", t)
    if m:
        return m.group(0).replace("cc", "").strip(" -").replace(" ", "-")
    return None


def license_accepted(slug: str | None, cfg: dict) -> bool:
    if slug is None:
        return False
    sec = cfg["security"]
    if slug in sec["rejected_license_slugs"]:
        return False
    return slug in sec["accepted_license_slugs"]


# --------------------------------------------------------------------------- #
# Source: PubMed Central OA subset via E-utilities (per-article license gate)
# --------------------------------------------------------------------------- #
def ingest_pmc(cfg: dict, raw_dir: Path, min_chars: int) -> int:
    import os
    import time
    import requests
    import defusedxml.ElementTree as ET  # XXE / billion-laughs safe

    src = cfg["sources"]["pmc"]
    if not src.get("enabled"):
        return 0
    if not src.get("email"):
        print("[pmc] REFUSING to run without `email` set (required by NCBI usage policy).")
        return 0

    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    api_key = os.environ.get(src.get("api_key_env", ""), "")
    delay = float(src.get("request_delay_s", 0.4))
    common = {"tool": "gray-matter", "email": src["email"]}
    if api_key:
        common["api_key"] = api_key

    def _get(endpoint, params):
        time.sleep(delay)  # respect NCBI rate policy
        r = requests.get(f"{base}/{endpoint}", params={**common, **params}, timeout=30)
        r.raise_for_status()
        return r

    written = 0
    for query in src.get("queries", []):
        try:
            ids = _get("esearch.fcgi", {
                "db": "pmc", "term": f"{query} AND open access[filter]",
                "retmax": src.get("retmax_per_query", 20), "retmode": "json",
            }).json()["esearchresult"]["idlist"]
        except Exception as e:
            print(f"[pmc] esearch failed for {query!r}: {e}")
            continue

        for pmcid in tqdm(ids, desc=f"[pmc] {query[:28]}"):
            try:
                xml = _get("efetch.fcgi", {"db": "pmc", "id": pmcid, "retmode": "xml"}).text
                root = ET.fromstring(xml)
            except Exception as e:
                print(f"[pmc] efetch/parse failed for PMC{pmcid}: {e}")
                continue

            # License gate: find any element mentioning a license, resolve its slug.
            slug = None
            for el in root.iter():
                tag = el.tag.lower()
                if "license" in tag:
                    blob = " ".join(filter(None, [
                        el.get("{http://www.w3.org/1999/xlink}href"),
                        el.get("href"), "".join(el.itertext()),
                    ]))
                    slug = license_slug(blob)
                    if slug:
                        break
            if not license_accepted(slug, cfg):
                continue  # ND / unknown / rejected -> drop, no ingestion

            title = " ".join((root.findtext(".//article-title") or f"PMC{pmcid}").split())
            body = root.find(".//body")
            if body is None:
                continue
            text = strip_html(" ".join(body.itertext()), cfg["security"]["strip_html_regex"])
            rec = {
                "title": title[:200],
                "text": text,
                "source": src["name"],
                "pmcid": f"PMC{pmcid}",
                "license": f"CC {slug.upper()}",
                "attribution": f"PMC{pmcid}, PubMed Central Open Access Subset, CC {slug.upper()}",
            }
            written += int(write_doc(raw_dir, rec, min_chars))
    print(f"[pmc] wrote {written} docs")
    return written


# --------------------------------------------------------------------------- #
# Source: LibreTexts Medicine — URL list, robots-compliant crawl delay
# --------------------------------------------------------------------------- #
def ingest_libretexts(cfg: dict, raw_dir: Path, min_chars: int) -> int:
    import time
    import requests

    src = cfg["sources"]["libretexts"]
    if not src.get("enabled"):
        return 0
    urls = src.get("urls") or []
    if not urls:
        print("[libretexts] enabled but no urls listed. Skipping.")
        return 0

    delay = float(src.get("crawl_delay_s", 5))  # required by robots.txt
    written = 0
    for url in tqdm(urls, desc="[libretexts] pages"):
        try:
            time.sleep(delay)
            resp = requests.get(url, timeout=30, headers={"User-Agent": "gray-matter/edu"})
            resp.raise_for_status()
        except Exception as e:
            print(f"[libretexts] skip {url}: {e}")
            continue
        text = strip_html(resp.text, cfg["security"]["strip_html_regex"])
        rec = {
            "title": url,
            "text": text,
            "source": src["name"],
            "license": src["license"],
            "attribution": src["attribution"],
        }
        written += int(write_doc(raw_dir, rec, min_chars))
    print(f"[libretexts] wrote {written} docs")
    return written


# --------------------------------------------------------------------------- #
def main() -> int:
    if not CONFIG_PATH.exists():
        print(f"Missing {CONFIG_PATH}. Run from the repo root.")
        return 1
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    if cfg["security"].get("trust_remote_code") is not False:
        print("SECURITY: security.trust_remote_code must be false. Aborting.")
        return 1

    raw_dir = Path(cfg["paths"]["corpus_raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    min_chars = int(cfg["ingest"]["min_doc_chars"])

    total = 0
    total += ingest_openstax(cfg, raw_dir, min_chars)
    total += ingest_medquad(cfg, raw_dir, min_chars)
    total += ingest_pmc(cfg, raw_dir, min_chars)
    total += ingest_libretexts(cfg, raw_dir, min_chars)
    print(f"\nDone. {total} new docs in {raw_dir} "
          f"({len(list(raw_dir.glob('*.json')))} total).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
