import copy
import hashlib
import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx

from benchtop import arxiv, study
from benchtop.env import ENVS, STATE

ARCHIVE_ID = "candidate_metadata_v1"
PAGE_SIZE = 100
OPEN_SEARCH = "http://a9.com/-/spec/opensearch/1.1/"
ID = re.compile(r"[0-9]{4}\.[0-9]{4,5}")


def now():
    return datetime.now(UTC)


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("metadata timestamps must be UTC strings")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("metadata timestamps must be explicit UTC")
    return parsed


def query(day, start):
    stamp = date.fromisoformat(day).strftime("%Y%m%d")
    categories = study._json(study.SPEC)["fresh_cohort"]["primary_categories"]
    return {"search_query": "(" + " OR ".join(f"cat:{category}" for category in categories)
            + f") AND submittedDate:[{stamp}0000 TO {stamp}2359]",
            "sortBy": "submittedDate", "sortOrder": "ascending", "start": start, "max_results": PAGE_SIZE}


def history():
    ids, sources = set(), {}
    for path in sorted(ENVS.glob("*/manifest.json")):
        value = study._json(path)
        if value.get("source") == "arxiv":
            ids.add(value["arxiv_id"])
            sources[str(path.resolve())] = study.cache.file_hash(path)
    funnel = STATE / "pipeline/funnel.jsonl"
    if funnel.exists():
        for line in funnel.read_text().splitlines():
            if line.strip():
                ids.add(json.loads(line)["id"])
        sources[str(funnel.resolve())] = study.cache.file_hash(funnel)
    for path in (STATE / "pipeline/drafts").glob("*"):
        match = re.match(r"^(\d{4})-(\d{4,5})-", path.name)
        if match:
            ids.add(".".join(match.groups()))
    if any(not isinstance(value, str) or ID.fullmatch(value) is None for value in ids):
        raise ValueError("project history contains an invalid paper ID")
    return {"ids": sorted(ids, key=lambda value: tuple(map(int, value.split(".")))), "sources": sources}


def exclusions(value):
    if (not isinstance(value, dict) or set(value) != {"schema_version", "complete_prior_history", "additional_ids"}
            or value["schema_version"] != 1 or value["complete_prior_history"] is not True
            or not isinstance(value["additional_ids"], list)):
        raise ValueError("attest complete prior manual-review/generation history and declare additional exclusions")
    seen = set()
    for row in value["additional_ids"]:
        if (not isinstance(row, dict) or set(row) != {"id", "reason"} or not isinstance(row["id"], str)
                or ID.fullmatch(row["id"]) is None or row["id"] in seen
                or not isinstance(row["reason"], str) or not row["reason"].strip()):
            raise ValueError("additional exclusions need unique versionless IDs and reasons")
        seen.add(row["id"])
    return copy.deepcopy(value)


def _bytes(path, raw):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as out:
        temporary = Path(out.name)
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse(raw, day, start):
    if len(raw) > 20 * 1024 * 1024:
        raise ValueError("metadata response exceeds the bounded archive size")
    try:
        feed = ET.fromstring(raw)
    except ET.ParseError as error:
        raise ValueError("invalid Atom metadata response") from error
    if feed.tag != "{http://www.w3.org/2005/Atom}feed":
        raise ValueError("expected an Atom metadata feed")
    total = int(feed.findtext(f"{{{OPEN_SEARCH}}}totalResults", "-1"))
    offset = int(feed.findtext(f"{{{OPEN_SEARCH}}}startIndex", "-1"))
    size = int(feed.findtext(f"{{{OPEN_SEARCH}}}itemsPerPage", "-1"))
    entries = feed.findall("a:entry", arxiv.ATOM)
    if (not 0 <= total <= 50000 or offset != start or not 0 <= size <= PAGE_SIZE
            or len(entries) != min(PAGE_SIZE, max(0, total - start)) or size < len(entries)):
        raise ValueError("incomplete, oversized or inconsistent metadata pagination; retry the same date")
    rows = []
    for node in entries:
        try:
            row = arxiv._entry(node)
        except (AttributeError, TypeError, IndexError) as error:
            raise ValueError("incomplete paper metadata") from error
        raw_id = node.findtext("a:id", namespaces=arxiv.ATOM)
        if (ID.fullmatch(row["id"]) is None or re.fullmatch(re.escape(row["id"]) + r"v[1-9][0-9]*", row["version"]) is None
                or raw_id not in (f"http://arxiv.org/abs/{row['version']}", f"https://arxiv.org/abs/{row['version']}")
                or timestamp(row["published"]).date().isoformat() != day
                or not row["title"] or not row["abstract"] or row["primary"] not in row["categories"]):
            raise ValueError("metadata has an unversioned, wrong-date or malformed paper")
        row["updated"] = node.findtext("a:updated", namespaces=arxiv.ATOM)
        if timestamp(row["updated"]) < timestamp(row["published"]):
            raise ValueError("paper update precedes first publication")
        rows.append(row)
    return total, rows


def _page(root, day, start, client, config, *, fresh=False):
    prefix = f"{day}-{start:06d}"
    for path in sorted((root / "requests").glob(prefix + "-*.json")):
        receipt = study._json(path)
        if not fresh and receipt.get("status") == "passed":
            return receipt
    attempts = [int(path.stem.rsplit("-", 1)[1]) for path in (root / "requests").glob(prefix + "-*.*")]
    index = max(attempts, default=-1) + 1
    stem = f"{prefix}-{index:04d}"
    receipt = {"day": day, "query": query(day, start), "endpoint": arxiv.API,
               "started_at_utc": now().isoformat(), "archive_configuration_sha256": config["configuration_sha256"],
               "status": "incomplete"}
    try:
        arxiv._polite()
        response = client.get(arxiv.API, params=receipt["query"], headers=arxiv.UA, timeout=60)
        receipt["http_status"] = response.status_code
        raw = response.content
        _bytes(root / "requests" / f"{stem}.xml", raw)
        receipt.update(raw_file=f"requests/{stem}.xml", raw_sha256=hashlib.sha256(raw).hexdigest())
        response.raise_for_status()
        parse(raw, day, start)
        receipt["status"] = "passed"
    except (httpx.HTTPError, ValueError, ET.ParseError, AttributeError, TypeError) as error:
        receipt.update(status="failed", error_type=type(error).__name__)
        raise ValueError("metadata retrieval failed; resume this archive on the same date before advancing") from error
    finally:
        receipt["finished_at_utc"] = now().isoformat()
        study._publish(root / "requests" / f"{stem}.json", study._seal(receipt, "receipt_sha256"))
    return study._json(root / "requests" / f"{stem}.json")


def _selection(root, config, pages, *, at):
    study._verify(config, "configuration_sha256")
    if (config["archive_id"] != ARCHIVE_ID or config["protocol_hash"] != study.canonical_hash(study._json(study.SPEC))
            or config["source_sha256"] != study._sources()
            or config["runtime"] != json.loads(json.dumps(study.cache.runtime_identity()))
            or config["latest_completed_date"] != (at.date() - timedelta(days=1)).isoformat()):
        raise ValueError("metadata archive is stale or belongs to another protocol/runtime/source snapshot")
    declared = exclusions(config["exclusions"])
    excluded = set(config["history"]["ids"]) | {row["id"] for row in declared["additional_ids"]}
    categories = study._json(study.SPEC)["fresh_cohort"]["primary_categories"]
    day, offset, expected_total, rows = date.fromisoformat(config["latest_completed_date"]), 0, None, {}
    seen, completed = set(), []
    for position, name in enumerate(pages):
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or name in seen:
            raise ValueError("metadata receipt path escapes the archive or repeats a page")
        seen.add(name)
        receipt = study._json(path)
        study._verify(receipt, "receipt_sha256")
        raw_path = (root / receipt["raw_file"]).resolve()
        if (receipt["status"] != "passed" or receipt["http_status"] != 200 or receipt["endpoint"] != arxiv.API
                or receipt["day"] != day.isoformat() or receipt["query"] != query(day.isoformat(), offset)
                or receipt["archive_configuration_sha256"] != config["configuration_sha256"]
                or not raw_path.is_relative_to(root.resolve()) or study.cache.file_hash(raw_path) != receipt["raw_sha256"]
                or not timestamp(config["started_at_utc"]) <= timestamp(receipt["started_at_utc"])
                <= timestamp(receipt["finished_at_utc"]) <= at):
            raise ValueError("metadata request/response provenance differs from its archive")
        total, batch = parse(raw_path.read_bytes(), day.isoformat(), offset)
        if expected_total is not None and expected_total != total:
            raise ValueError("metadata result count changed between pages; retry the same date")
        expected_total = total
        for row in batch:
            if row["id"] in rows and row != rows[row["id"]]:
                raise ValueError("conflicting fetched paper versions or metadata; never choose a version silently")
            rows[row["id"]] = row
        offset += len(batch)
        if offset >= total:
            eligible = [row for row in rows.values() if row["primary"] in categories and row["id"] not in excluded]
            completed.append({"date": day.isoformat(), "total_entries": total, "unseen_primary_papers": len(eligible)})
            if eligible:
                if position != len(pages) - 1:
                    raise ValueError("archive skipped a newer eligible date")
                ordered = sorted(eligible, key=lambda row: tuple(map(int, row["id"].split("."))))
                return {"selection_date": day.isoformat(), "dates_checked": completed,
                        "candidates": ordered[:study._json(study.SPEC)["fresh_cohort"]["max_candidate_papers"]]}
            day, offset, expected_total, rows = day - timedelta(days=1), 0, None, {}
    raise ValueError("archive has no complete eligible date; resume metadata retrieval")


def collect(directory, declaration, *, client=None):
    from benchtop.session_ledger import _writable

    root = _writable(directory)
    at = now()
    declared = exclusions(declaration)
    current = {"archive_id": ARCHIVE_ID, "protocol_hash": study.canonical_hash(study._json(study.SPEC)),
               "source_sha256": study._sources(), "runtime": json.loads(json.dumps(study.cache.runtime_identity())),
               "exclusions": declared, "history": history()}
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    from benchtop.primary import _guard

    with _guard(root / ".archive.lock"):
        config_path = root / "configuration.json"
        if config_path.exists():
            config = study._json(config_path)
            study._verify(config, "configuration_sha256")
            if any(config[key] != value for key, value in current.items()):
                raise ValueError("archive configuration changed; preserve the old attempt")
        else:
            if any(path.name != ".archive.lock" for path in root.iterdir()):
                raise ValueError("choose an empty metadata archive directory; unrelated files are preserved")
            config = study._seal({**current, "started_at_utc": at.isoformat(),
                                 "latest_completed_date": (at.date() - timedelta(days=1)).isoformat()}, "configuration_sha256")
            study._publish(config_path, config)
        if (root / "archive.json").exists():
            return load(root, at=at)
        (root / "requests").mkdir(exist_ok=True, mode=0o700)
        owned = client is None
        client = client or httpx.Client()
        pages = []
        try:
            for days in range(31):
                day = (date.fromisoformat(config["latest_completed_date"]) - timedelta(days=days)).isoformat()
                start, expected_total = 0, None
                while True:
                    restart = root / "requests" / f"restart-{day}.json"
                    receipt = _page(root, day, start, client, config, fresh=restart.exists())
                    path = next(path for path in (root / "requests").glob(f"{day}-{start:06d}-*.json")
                                if study._json(path) == receipt)
                    pages.append(str(path.relative_to(root)))
                    total, rows = parse((root / receipt["raw_file"]).read_bytes(), day, start)
                    if expected_total is not None and total != expected_total:
                        if not restart.exists():
                            study._publish(restart, {"day": day, "reason": "inconsistent_pagination", "recorded_at_utc": now().isoformat()})
                        raise ValueError("metadata count changed; do not advance dates")
                    expected_total = total
                    start += len(rows)
                    if start >= total:
                        break
                try:
                    selected = _selection(root, config, pages, at=now())
                except ValueError as error:
                    if str(error) != "archive has no complete eligible date; resume metadata retrieval":
                        raise
                else:
                    archive = study._seal({"schema_version": 1, "archive_id": ARCHIVE_ID,
                        "configuration_sha256": config["configuration_sha256"], "pages": pages, **selected,
                        "request_evidence": {str(path.relative_to(root)): study.cache.file_hash(path)
                                             for path in sorted((root / "requests").iterdir()) if path.is_file()},
                        "model_sessions": 0, "completed_at_utc": now().isoformat()}, "archive_sha256")
                    study._publish(root / "archive.json", archive)
                    return load(root, at=now())
            raise ValueError("no eligible date in 31 days; preserve the archive and investigate before extending retrieval")
        finally:
            if owned:
                client.close()


def load(directory, *, at=None):
    root = Path(directory).resolve()
    at = at or now()
    archive, config = study._json(root / "archive.json"), study._json(root / "configuration.json")
    study._verify(archive, "archive_sha256")
    if (archive.get("archive_id") != ARCHIVE_ID or archive.get("schema_version") != 1
            or archive["configuration_sha256"] != config["configuration_sha256"] or archive.get("model_sessions") != 0
            or timestamp(archive["completed_at_utc"]) > at):
        raise ValueError("invalid metadata archive binding")
    actual = {str(path.relative_to(root)): study.cache.file_hash(path)
              for path in sorted((root / "requests").iterdir()) if path.is_file()}
    if any(not path.is_file() for path in (root / "requests").iterdir()):
        raise ValueError("unexpected directory in metadata request history")
    if archive.get("request_evidence") != actual:
        raise ValueError("metadata request history changed, including failed or interrupted retrievals")
    expected = _selection(root, config, archive["pages"], at=at)
    if any(archive[key] != value for key, value in expected.items()):
        raise ValueError("candidate order/date/versions differ from the archived complete metadata")
    return archive
