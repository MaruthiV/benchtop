import copy
from datetime import UTC, datetime


def _context():
    from benchtop import primary, study

    spec = study._json(study.SPEC)
    return primary, study, spec, spec.get("continuation")


def _path(primary, parent_hash):
    return primary.STATE_BASE / "continuations" / (parent_hash + ".json")


def _inventory(root):
    from benchtop.cache import file_hash

    return {str(path.relative_to(root)): file_hash(path) for path in sorted(root.rglob("*"))
            if path.is_file() and not path.name.startswith(".")}


def _limits(spec, old, declared):
    if set(spec) != set(old) | {"continuation"}:
        raise ValueError("continuation has unexpected protocol fields")
    same = set(old) - {"protocol_id", "protocol_version", "status", "binding_policy", "development", "agent", "execution_limits"}
    if any(spec.get(key) != old[key] for key in same):
        raise ValueError("continuation changes the scientific protocol beyond the declared smoke correction")
    for section, excluded in (("binding_policy", "path"), ("development", "smoke_rule"), ("agent", "prompt_binding")):
        if {key: value for key, value in spec[section].items() if key != excluded} != {key: value for key, value in old[section].items() if key != excluded}:
            raise ValueError("continuation changes the frozen scientific rules")
    previous, current = old["execution_limits"], spec["execution_limits"]
    expected = copy.deepcopy(previous)
    expected["total_max_episode_starts"] -= declared["parent_expected_start_intents"]
    expected["reserve"]["infrastructure_retries"] -= declared["parent_expected_start_intents"]
    if (declared["global_total_max_episode_starts"] != previous["total_max_episode_starts"]
            or current != expected or current["reserve"]["infrastructure_retries"] < 0):
        raise ValueError("continuation must deduct prior consumption without increasing any allowance")


def _parent():
    from benchtop import calibrate
    from benchtop import session_ledger as ledger
    from benchtop.cache import file_hash

    primary, study, spec, declared = _context()
    if not declared:
        return None
    root = primary.STATE_BASE / "study" / declared["parent_protocol_hash"]
    original = study.ROOT / "benchmarks/release_v1/spec.json"
    if file_hash(original) != declared["parent_protocol_file_sha256"]:
        raise ValueError("original protocol bytes changed")
    original_references = study.ROOT / "benchmarks/release_v1/admission/references/selection.json"
    references = study._json(study.SPEC.parent / "references.json")
    if (references["protocol_sha256"] != file_hash(study.SPEC)
            or references["cases"] != study._json(original_references)["cases"]
            or references["continuation_provenance"]["original_manifest_sha256"] != file_hash(original_references)):
        raise ValueError("continuation must preserve the original ordered reference cases")
    with ledger._locked(root / "ledger") as (directory, manifest):
        state = ledger._read(directory, manifest)
        old = manifest["identity"]["protocol"]
        _limits(spec, old, declared)
        counts = dict(state["counts"])
        if (manifest["identity"]["scope"] != "primary" or manifest["ledger_id"] != declared["parent_ledger_id"]
                or manifest["identity"]["protocol_hash"] != declared["parent_protocol_hash"]
                or file_hash(directory / "ledger.json") != declared["parent_manifest_sha256"]
                or study.canonical_hash(_inventory(root)) != declared["parent_inventory_sha256"]
                or counts["total"] != declared["parent_expected_start_intents"] or counts["smoke"] != counts["total"]
                or counts["active"] or counts["primary"] or counts["integration"] or counts["retry"] or state["halted"]):
            raise ValueError("parent source epoch or consumed usage changed; continuation is blocked")
        slots = [{"attempts": ledger._normalized_history(history)} for history in state["histories"].values()]
        return {"parent_protocol_hash": declared["parent_protocol_hash"], "parent_ledger_id": manifest["ledger_id"],
                "parent_manifest_sha256": declared["parent_manifest_sha256"], "parent_inventory_sha256": declared["parent_inventory_sha256"],
                "prior_counts": counts, "prior_costs": calibrate._costs(slots),
                "parent_model_configuration": manifest["identity"]["model_configuration"],
                "remaining_start_intents": spec["execution_limits"]["total_max_episode_starts"],
                "global_total_max_episode_starts": declared["global_total_max_episode_starts"]}


def guard(manifest):
    primary, study, spec, _ = _context()
    if manifest["identity"]["scope"] != "primary":
        return
    identity = manifest["identity"]["protocol_hash"]
    if identity != study.canonical_hash(spec):
        raise ValueError("active protocol differs from the canonical journal")
    record = verify()
    if record and manifest["identity"]["model_configuration"] != record["parent_model_configuration"]:
        raise ValueError("continuation model differs from the retained parent")


def prepare():
    primary, study, spec, declared = _context()
    if not declared:
        verify()
        return None
    parent = _parent()
    path = _path(primary, declared["parent_protocol_hash"])
    if path.exists():
        return verify()
    record = study._seal({"schema_version": 1, "id": "smoke_prompt_continuation_v1", **parent,
        "successor_protocol_hash": study.canonical_hash(spec), "successor_spec_sha256": study.cache.file_hash(study.SPEC),
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "scope": "Retire the unchanged parent from new starts. Keep original evidence/costs and deduct prior consumption from the successor. Parent smoke is excluded from primary model outcomes."}, "record_sha256")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    study._publish(path, record)
    return record


def verify():
    primary, study, spec, declared = _context()
    if not declared:
        if _path(primary, study.canonical_hash(spec)).exists():
            raise ValueError("this protocol is retired; retain its records and use the declared continuation")
        return None
    parent = _parent()
    record = study._json(_path(primary, declared["parent_protocol_hash"]))
    study._verify(record, "record_sha256")
    if (record.get("id") != "smoke_prompt_continuation_v1" or any(record.get(key) != value for key, value in parent.items())
            or record.get("successor_protocol_hash") != study.canonical_hash(spec)
            or record.get("successor_spec_sha256") != study.cache.file_hash(study.SPEC)):
        raise ValueError("continuation authority differs from the parent usage or successor protocol")
    return record


def usage():
    _, _, _, declared = _context()
    if not declared:
        return None
    record = verify()
    return {key: record[key] for key in ("parent_protocol_hash", "parent_ledger_id", "prior_counts", "prior_costs",
        "remaining_start_intents", "global_total_max_episode_starts", "record_sha256")} | {
        "interpretation": "Excluded earlier compatibility smoke, retained as consumed usage and reported overhead; never pooled into successor primary outcomes."}
