import copy
import json
from collections import Counter

import pytest

from benchtop import replay
from benchtop.agents.baselines import play_release
from benchtop.env import ROOT
from benchtop.episode import Episode
from tests.test_admission import human_control
from tests.test_coverage_policies import fixture


def make_plan(tmp_path, repeats=1):
    env = fixture(tmp_path)
    return env, replay.make_plan([env.dir], repeats=repeats)


def finishes(directory):
    return {path.name: path.read_bytes() for path in (directory / "attempts").glob("*.finish.json")}


def test_smoke_grid_is_bounded_versioned_and_json_roundtrippable(tmp_path):
    _, plan = make_plan(tmp_path)
    assert len(plan["slots"]) == 15
    assert len({slot["slot_id"] for slot in plan["slots"]}) == 15
    assert plan == json.loads(json.dumps(plan))
    assert "development_majority_label" not in plan["binding"]["policies"]
    assert plan["binding"]["development_gate"] is None
    assert set(plan["binding"]["environments"][0]["fingerprint"]["visible_order"]) == {0, 1, 2, 3}
    with pytest.raises(ValueError, match="one or two"):
        replay.make_plan([tmp_path / "coverage"], repeats=200)


def test_raw_replay_resume_never_replaces_a_completed_wrong_answer(tmp_path):
    _, plan = make_plan(tmp_path)
    directory = tmp_path / "run"
    first = replay.run_plan(plan, directory, max_attempts=1)
    assert first["slots"][0]["status"] == "completed" and first["slots"][0]["reward"] == 0
    before = finishes(directory)
    full = replay.run_plan(plan, directory, retry_technical=True)
    assert full["attempts_invoked_this_run"] == 14
    assert all(finishes(directory)[name] == contents for name, contents in before.items())
    assert replay.run_plan(plan, directory, retry_technical=True)["attempts_invoked_this_run"] == 0
    assert sum(group["planned"] for group in full["groups"]) == 15
    record = json.loads((directory / "attempts" / f"{replay._attempt_id(plan['slots'][0], 0)}.finish.json").read_text())
    assert record["tool_trace"][0]["response"]["results"]
    assert record["evaluator_log"]
    assert record["outcome"]["visible_order_source"] == "explicit"


def test_technical_errors_remain_missing_with_at_most_one_explicit_retry(tmp_path, monkeypatch):
    _, plan = make_plan(tmp_path)
    directory = tmp_path / "run"
    def unavailable(self, configs, seeds=1):
        raise OSError("Controlled unavailable experiment service")

    monkeypatch.setattr(Episode, "run", unavailable)
    initial = replay.run_plan(plan, directory)
    # Priors plus the two too-small-for-a-pair coverage slots never contact the failed service.
    assert Counter(slot["status"] for slot in initial["slots"]) == {"technical_failure": 10, "completed": 5}
    assert replay.run_plan(plan, directory)["attempts_invoked_this_run"] == 0
    assert replay.run_plan(plan, directory, retry_technical=True)["attempts_invoked_this_run"] == 10
    final = replay.run_plan(plan, directory, retry_technical=True)
    assert final["attempts_invoked_this_run"] == 0
    assert all(slot["reward"] is None for slot in final["slots"] if slot["status"] == "technical_failure")
    assert all(group["complete_case_accuracy"] is None for group in final["groups"] if group["technical_missing"])
    record = next(json.loads(path.read_text()) for path in (directory / "attempts").glob("*.finish.json")
                  if json.loads(path.read_text())["status"] == "technical_failure")
    assert record["tool_trace"][0]["exception"]["type"] == "OSError"


def test_invalid_policy_request_is_zero_not_a_retryable_service_failure(tmp_path, monkeypatch):
    _, plan = make_plan(tmp_path)
    directory = tmp_path / "run"
    original = Episode.run
    def invalid(self, configs, seeds=1):
        return {"error": "Controlled invalid request"}

    monkeypatch.setattr(Episode, "run", invalid)
    replay.run_plan(plan, directory, max_attempts=1)
    monkeypatch.setattr(Episode, "run", original)
    before = finishes(directory)
    result = replay.run_plan(plan, directory, retry_technical=True, max_attempts=0)
    assert result["slots"][0]["status"] == "policy_failure"
    assert result["slots"][0]["reward"] == 0
    assert finishes(directory) == before


def test_interrupted_attempt_preserves_unknown_cost_and_is_recoverable(tmp_path):
    _, plan = make_plan(tmp_path)
    directory = tmp_path / "run"
    replay.run_plan(plan, directory, max_attempts=0)
    slot = plan["slots"][0]
    aid = replay._attempt_id(slot, 0)
    start = replay._sealed({"plan_id": plan["plan_id"], "slot_id": slot["slot_id"], "attempt_id": aid,
                            "attempt_index": 0, "started_at_utc": replay._now()}, "record_sha256")
    replay._publish(directory / "attempts" / f"{aid}.start.json", start)
    assert replay.summarize(plan, directory)["slots"][0]["status"] == "interrupted"
    missing = replay.run_plan(plan, directory, max_attempts=0)
    assert missing["slots"][0]["reward"] is None
    assert sum(group["unknown_spend_attempts"] for group in missing["groups"]) == 1
    resumed = replay.run_plan(plan, directory, retry_technical=True, max_attempts=1)
    assert resumed["slots"][0]["status"] == "completed"
    assert sum(group["unknown_spend_attempts"] for group in resumed["groups"]) == 1


def test_tampered_or_incomplete_grid_and_record_fail_visibly(tmp_path):
    _, plan = make_plan(tmp_path)
    changed = copy.deepcopy(plan)
    changed["slots"].pop()
    changed = replay._sealed(changed, "plan_id")
    with pytest.raises(ValueError, match="full bound grid"):
        replay.run_plan(changed, tmp_path / "bad")
    directory = tmp_path / "run"
    replay.run_plan(plan, directory, max_attempts=1)
    path = next((directory / "attempts").glob("*.finish.json"))
    value = json.loads(path.read_text())
    value["reward"] = 1.0
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="record_sha256"):
        replay.summarize(plan, directory)


def test_changed_implementation_or_cache_cannot_resume(tmp_path, monkeypatch):
    env, plan = make_plan(tmp_path)
    directory = tmp_path / "run"
    replay.run_plan(plan, directory, max_attempts=1)
    before = finishes(directory)
    with monkeypatch.context() as patch:
        original = replay.cache.file_hash
        patch.setattr(replay.cache, "file_hash", lambda path: "0"*64 if str(path).endswith("agents/baselines.py") else original(path))
        with pytest.raises(ValueError, match="implementation changed"):
            replay.run_plan(plan, directory)
    with (env.dir / "lab.py").open("a") as out:
        out.write("\n# Deliberate test-only source change\n")
    with pytest.raises(ValueError):
        replay.run_plan(plan, directory)
    assert finishes(directory) == before


def test_prior_and_random_controls_use_the_intended_information(tmp_path):
    env = fixture(tmp_path)
    prior = play_release(Episode(env, "claim", seed=0), "paper_prior_only")
    assert prior["calls"] == 0 and prior["spent"] == 0 and prior["estimator_id"] is None
    with pytest.raises(ValueError, match="bound label"):
        play_release(Episode(env, "claim", seed=0), "development_majority_label")
    majority = play_release(Episode(env, "claim", seed=0), "development_majority_label", majority_label="baseline_better")
    assert majority["answer"] == "baseline_better" and majority["calls"] == 0
    random = play_release(Episode(env, "claim", seed=0, budget_frac=1.0), "random_experiment_design", seed=7)
    assert random["estimator_id"] == "paired_context_v1"
    assert random["spent"] <= random["budget"] and len(random["tool_trace"]) == random["calls"]


def test_explicit_visible_order_is_private_and_cannot_include_heldout_ids(tmp_path):
    env = fixture(tmp_path)
    ep = Episode(env, "claim", seed=99, visible_order=[3, 1, 0, 2])
    assert ep.summary()["visible_order"] == [3, 1, 0, 2]
    assert "visible_order" not in ep.describe()
    with pytest.raises(ValueError, match="permutation"):
        Episode(env, "claim", visible_order=[0, 1, 2, 4])


def test_real_unassigned_records_do_not_authorize_development_replay():
    gate = replay.development_gate(ROOT / "benchmarks/release_v1/admission/references/drafts")
    assert gate["status"] == "blocked" and gate["selected_env_ids"] == []
    with pytest.raises(ValueError, match="development replay blocked"):
        replay.make_plan(mode="development", reviews_dir=ROOT / "benchmarks/release_v1/admission/references/drafts")


def test_development_membership_and_majority_are_bound_to_six_simulated_reviews(tmp_path):
    """Temporary simulated identities test the gate; they are not actual reference reviews."""
    from benchtop.admission import inspect_lab
    ids = [row["env"] for row in json.loads(replay.REFERENCES.read_text())["cases"]]
    for env in ids:
        record = human_control(inspect_lab(ROOT / "envs" / env))
        (tmp_path / f"{env}.json").write_text(json.dumps(record))
    gate = replay.development_gate(tmp_path)
    assert gate["status"] == "ready" and gate["selected_env_ids"] == ids[:4]
    assert gate["majority_label"] == "paper_better"
    plan = replay.make_plan(mode="development", reviews_dir=tmp_path)
    assert len(plan["slots"]) == 4 * 6 * 3 * 200
    assert plan["binding"]["majority_label"] == gate["majority_label"]
    # A changed/stale review must not leave the earlier plan authorized.
    path = tmp_path / f"{ids[0]}.json"
    record = json.loads(path.read_text())
    record["subject"]["lab_sha256"] = "0"*64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="admission changed"):
        replay.run_plan(plan, tmp_path / "unused", reviews_dir=tmp_path, max_attempts=0)


def test_original_output_paths_and_existing_outcomes_are_protected(tmp_path):
    _, plan = make_plan(tmp_path)
    with pytest.raises(ValueError, match="under artifacts"):
        replay.run_plan(plan, ROOT / "results")
    path = tmp_path / "immutable.json"
    replay._publish(path, {"value": 0})
    with pytest.raises(FileExistsError):
        replay._publish(path, {"value": 1})
    assert json.loads(path.read_text()) == {"value": 0}


def test_majority_ties_use_the_protocol_label_order():
    order = ["paper_better", "baseline_better", "no_meaningful_difference"]
    assert replay._majority([order[0], order[1], order[0], order[1]], order) == order[0]
    assert replay._majority([order[1], order[2], order[1], order[2]], order) == order[1]


def test_another_runner_cannot_share_the_output_directory(tmp_path):
    _, plan = make_plan(tmp_path)
    directory = tmp_path / "run"
    directory.mkdir()
    with (directory / ".replay.lock").open("a") as lock:
        replay.fcntl.flock(lock, replay.fcntl.LOCK_EX | replay.fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="already running"):
            replay.run_plan(plan, directory)
    assert not (directory / "plan.json").exists()
