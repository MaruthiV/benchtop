import json
from pathlib import Path

from benchtop.env import Env
from benchtop.release_data import compare
from benchtop.tasks import build_task_set

root = Path(__file__).resolve().parents[1]
summary = json.loads((root / "qualification-only/summary.json").read_text())
for row in summary:
    folder = root / "qualification-only" / row["version"]
    env = Env.load(folder, replay=True)
    allowed = env.visible_seeds + env.qualification_seeds
    assert all(seed in allowed for cells in env.table.cells.values() for seed in cells)
    assert row["confirmation_evaluated"] is False
    expected = json.loads((folder / "tasks.json").read_text())
    actual = build_task_set(env, seeds=env.qualification_seeds)
    compare(expected, actual)
    assert not any(task["id"] == "claim" for task in actual["tasks"])
    assert row["status"] == "qualification_ambiguous"
print(json.dumps({"qualification_only_labs_reproduced": len(summary), "hidden_claims_invented": 0, "models_started": 0, "lab_workers_started": 0}))
