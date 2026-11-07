import copy
import json
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from benchtop import candidate_metadata as metadata
from benchtop import study

AT = datetime(2026, 11, 7, 12, tzinfo=UTC)
DECLARATION = {"schema_version": 1, "complete_prior_history": True, "additional_ids": []}


def feed(day, rows, *, total=None, start=0):
    atom, arxiv = metadata.arxiv.ATOM["a"], metadata.arxiv.ATOM["arxiv"]
    root = ET.Element(f"{{{atom}}}feed")
    for name, value in (("totalResults", len(rows) if total is None else total), ("startIndex", start), ("itemsPerPage", len(rows))):
        ET.SubElement(root, f"{{{metadata.OPEN_SEARCH}}}{name}").text = str(value)
    for row in rows:
        entry = ET.SubElement(root, f"{{{atom}}}entry")
        for name, value in {"id": f"http://arxiv.org/abs/{row['id']}v{row.get('version', 1)}", "title": "Metadata fixture " + row["id"],
                            "summary": "Test fixture only; not a real paper.", "published": day + "T01:00:00Z",
                            "updated": day + "T02:00:00Z"}.items():
            ET.SubElement(entry, f"{{{atom}}}{name}").text = value
        author = ET.SubElement(entry, f"{{{atom}}}author")
        ET.SubElement(author, f"{{{atom}}}name").text = "Fixture Author"
        for category in {row.get("primary", "cs.LG"), "cs.LG"}:
            ET.SubElement(entry, f"{{{atom}}}category", term=category)
        ET.SubElement(entry, f"{{{arxiv}}}primary_category", term=row.get("primary", "cs.LG"))
    return ET.tostring(root)


class Client:
    def __init__(self, responses):
        self.responses, self.calls = responses, []

    def get(self, endpoint, *, params, **kwargs):
        query = params["search_query"]
        stamp = query.split("submittedDate:[")[1][:8]
        day = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}"
        key = day, params["start"]
        self.calls.append(key)
        value = self.responses[key]
        if isinstance(value, Exception):
            raise value
        status, body = value if isinstance(value, tuple) else (200, value)
        return httpx.Response(status, content=body, request=httpx.Request("GET", endpoint))


def controls(monkeypatch):
    monkeypatch.setattr(metadata, "now", lambda: AT)
    monkeypatch.setattr(metadata.arxiv, "_polite", lambda: None)
    monkeypatch.setattr(metadata, "history", lambda: {"ids": [], "sources": {}})


def test_numeric_first_16_excludes_prior_and_secondary_categories_and_retains_version(tmp_path, monkeypatch):
    controls(monkeypatch)
    rows = [{"id": f"2611.{index:05d}", "version": 2} for index in range(22, 0, -1)]
    rows[-2]["primary"] = "cs.AI"
    rows.append(copy.deepcopy(rows[18]))
    declared = {**DECLARATION, "additional_ids": [{"id": "2611.00001", "reason": "Previously reviewed test control"}]}
    client = Client({("2026-11-06", 0): feed("2026-11-06", rows)})
    result = metadata.collect(tmp_path, declared, client=client)
    assert [row["id"] for row in result["candidates"]] == [f"2611.{index:05d}" for index in range(3, 19)]
    assert all(row["version"] == row["id"] + "v2" for row in result["candidates"])
    assert result["model_sessions"] == 0 and len(client.calls) == 1
    assert metadata.collect(tmp_path, declared, client=Client({})) == result
    assert (tmp_path / "archive.json").stat().st_mode & 0o777 == 0o600


def test_empty_dates_are_archived_and_dates_cannot_be_skipped(tmp_path, monkeypatch):
    controls(monkeypatch)
    client = Client({("2026-11-06", 0): feed("2026-11-06", []),
                     ("2026-11-05", 0): feed("2026-11-05", [{"id": "2611.00003"}])})
    result = metadata.collect(tmp_path, DECLARATION, client=client)
    assert result["selection_date"] == "2026-11-05" and len(result["dates_checked"]) == 2
    altered = {**result, "pages": result["pages"][1:]}
    (tmp_path / "archive.json").write_text(json.dumps(study._seal(altered, "archive_sha256")))
    with pytest.raises(ValueError, match="provenance"):
        metadata.load(tmp_path)


def test_all_pages_are_required_before_candidate_selection(tmp_path, monkeypatch):
    controls(monkeypatch)
    rows = [{"id": f"2611.{index:05d}"} for index in range(1, 102)]
    client = Client({("2026-11-06", 0): feed("2026-11-06", rows[:100], total=101),
                     ("2026-11-06", 100): feed("2026-11-06", rows[100:], total=101, start=100)})
    result = metadata.collect(tmp_path, DECLARATION, client=client)
    assert client.calls == [("2026-11-06", 0), ("2026-11-06", 100)] and len(result["pages"]) == 2
    assert result["dates_checked"][0]["total_entries"] == 101


def test_failed_date_is_not_skipped_and_retry_preserves_failure_response(tmp_path, monkeypatch):
    controls(monkeypatch)
    first = Client({("2026-11-06", 0): (503, b"fixture service unavailable")})
    with pytest.raises(ValueError, match="same date"):
        metadata.collect(tmp_path, DECLARATION, client=first)
    assert first.calls == [("2026-11-06", 0)] and not (tmp_path / "archive.json").exists()
    failed = next((tmp_path / "requests").glob("*.json"))
    original = failed.read_bytes()
    second = Client({("2026-11-06", 0): feed("2026-11-06", [{"id": "2611.00001"}])})
    result = metadata.collect(tmp_path, DECLARATION, client=second)
    assert second.calls == first.calls and failed.read_bytes() == original
    assert study._json(failed)["status"] == "failed" and len(result["request_evidence"]) == 4
    (tmp_path / study._json(failed)["raw_file"]).write_bytes(b"changed failed response")
    with pytest.raises(ValueError, match="request history changed"):
        metadata.load(tmp_path)


def test_inconsistent_pagination_retries_whole_date_without_overwriting_prior_pages(tmp_path, monkeypatch):
    controls(monkeypatch)
    rows = [{"id": f"2611.{index:05d}"} for index in range(1, 103)]
    first = Client({("2026-11-06", 0): feed("2026-11-06", rows[:100], total=101),
                    ("2026-11-06", 100): feed("2026-11-06", rows[100:], total=102, start=100)})
    with pytest.raises(ValueError, match="count changed"):
        metadata.collect(tmp_path, DECLARATION, client=first)
    original = {str(path): path.read_bytes() for path in (tmp_path / "requests").iterdir()}
    second = Client({("2026-11-06", 0): feed("2026-11-06", rows[:100], total=102),
                     ("2026-11-06", 100): feed("2026-11-06", rows[100:], total=102, start=100)})
    result = metadata.collect(tmp_path, DECLARATION, client=second)
    assert result["dates_checked"][0]["total_entries"] == 102 and len(second.calls) == 2
    assert all(metadata.Path(name).read_bytes() == raw for name, raw in original.items())


@pytest.mark.parametrize("defect", ["xml", "truncated", "wrong_date", "conflicting_version"])
def test_malformed_responses_never_authorize_skipping_a_date(tmp_path, monkeypatch, defect):
    controls(monkeypatch)
    raw = feed("2026-11-06", [{"id": "2611.00001"}])
    if defect == "xml":
        raw = b"not XML"
    elif defect == "truncated":
        raw = feed("2026-11-06", [{"id": "2611.00001"}], total=2)
    elif defect == "wrong_date":
        raw = feed("2026-11-05", [{"id": "2611.00001"}])
    else:
        raw = feed("2026-11-06", [{"id": "2611.00001", "version": 1}, {"id": "2611.00001", "version": 2}])
    client = Client({("2026-11-06", 0): raw})
    with pytest.raises(ValueError):
        metadata.collect(tmp_path, DECLARATION, client=client)
    assert client.calls == [("2026-11-06", 0)] and not (tmp_path / "archive.json").exists()


def test_archive_and_exclusion_changes_are_visible(tmp_path, monkeypatch):
    controls(monkeypatch)
    client = Client({("2026-11-06", 0): feed("2026-11-06", [{"id": "2611.00001"}])})
    result = metadata.collect(tmp_path, DECLARATION, client=client)
    monkeypatch.setattr(metadata, "now", lambda: AT + timedelta(days=1))
    with pytest.raises(ValueError, match="stale"):
        metadata.load(tmp_path)
    assert metadata.load(tmp_path, at=AT) == result
    with pytest.raises(ValueError, match="configuration changed"):
        metadata.collect(tmp_path, {**DECLARATION, "additional_ids": [{"id": "2611.00002", "reason": "new review"}]}, client=client)


def test_history_includes_generated_rejected_and_draft_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(metadata, "ENVS", tmp_path / "envs")
    monkeypatch.setattr(metadata, "STATE", tmp_path / "private")
    manifest = tmp_path / "envs/lab/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"source": "arxiv", "arxiv_id": "2610.00001"}))
    funnel = tmp_path / "private/pipeline/funnel.jsonl"
    funnel.parent.mkdir(parents=True)
    funnel.write_text(json.dumps({"id": "2610.00002", "outcome": "synth_failed"}) + "\n")
    (funnel.parent / "drafts/2610-00003-test").mkdir(parents=True)
    assert metadata.history()["ids"] == ["2610.00001", "2610.00002", "2610.00003"]


def test_unrelated_destination_is_preserved(tmp_path, monkeypatch):
    controls(monkeypatch)
    (tmp_path / "unrelated.txt").write_text("preserve me")
    with pytest.raises(ValueError, match="empty metadata archive"):
        metadata.collect(tmp_path, DECLARATION, client=Client({}))
    assert (tmp_path / "unrelated.txt").read_text() == "preserve me"
