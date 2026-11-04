from benchtop.analysis import calibration, decompose, implied_answer
from benchtop.episode import Episode
from tests.test_core import toy_env


def play_and_row(env, answer, configs, seeds=2):
    ep = Episode(env, "claim", seed=1, budget_frac=1.0)
    ep.run(configs, seeds=seeds)
    ep.submit(answer, 0.8)
    ran = [pair for e in ep.log if e["action"] == "run" for pair in e["ran"]]
    return {
        "env": env.id,
        "task": "claim",
        "answer": answer,
        "score": ep.result["score"],
        "submitted": True,
        "ran": ran,
    }


def test_data_implied_answer(tmp_path):
    env = toy_env(tmp_path)
    both = [{"method": m, "ctx": c, "width": w} for m in ("new", "base") for c in (0, 1, 2) for w in (8, 64)]
    assert (
        implied_answer(env, "claim", play_and_row(env, "paper_better", both)["ran"])
        == "no_meaningful_difference"
    )
    only_gain = [{"method": m, "ctx": 2} for m in ("new", "base")]
    assert implied_answer(env, "claim", play_and_row(env, "paper_better", only_gain)["ran"]) == "paper_better"


def test_decompose_categories(tmp_path):
    env = toy_env(tmp_path)
    both = [{"method": m, "ctx": c, "width": w} for m in ("new", "base") for c in (0, 1, 2) for w in (8, 64)]
    only_gain = [{"method": m, "ctx": 2} for m in ("new", "base")]
    assert decompose(env, play_and_row(env, "no_meaningful_difference", both)) == "right"
    # data says no difference, agent sides with the paper anyway
    assert decompose(env, play_and_row(env, "paper_better", both)) == "sided_with_paper_over_own_data"
    # agent only looked where the method wins, followed that data, and was wrong
    assert decompose(env, play_and_row(env, "paper_better", only_gain)) == "followed_own_data_but_wrong"


def test_calibration():
    rows = [{"confidence": 0.9, "score": 1.0}, {"confidence": 0.9, "score": 0.0}]
    out = calibration(rows)
    assert out["accuracy"] == 0.5 and abs(out["overconfidence"] - 0.4) < 1e-9
