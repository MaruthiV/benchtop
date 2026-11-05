"""Shared visible-data reconstruction for versioned classical policies."""

import math

import numpy as np

from benchtop.space import Space, canon, config_key
from benchtop.stats import average_contrast, confidence_interval, interval_label, point_label
from benchtop.tasks import COMPONENT_LABELS, grid_over, sign

ESTIMATOR_ID = "paired_context_v1"


def arms(space, question):
    kind = question["kind"]
    if kind == "component":
        factor = space.factor(question["factor"])
        if len(factor.levels) != 2:
            raise ValueError("component reconstruction requires two levels")
        return factor.name, factor.levels[1], factor.levels[0]
    if kind not in ("claim", "where"):
        raise ValueError("shared estimator supports claim, where, and component questions")
    if kind == "where":
        space.factor(question["factor"])
        if question["factor"] == space.treatment:
            raise ValueError("where factor must differ from the treatment")
    return space.treatment, space.paper_level, space.baseline_level


def public_problem(description):
    """Construct the entire policy input from fields also delivered to an agent."""
    treatment = description["treatment"]
    raw = {
        "title": description["lab"], "setting": description.get("setting", ""),
        "metric": dict(description["metric"]),
        "factors": [{**factor, "role": "treatment" if factor["name"] == treatment["factor"] else "context"}
                    for factor in description["factors"]],
        "treatment": {"factor": treatment["factor"], "paper": treatment["paper_method"], "baseline": treatment["baseline"]},
    }
    question = {key: description["question_spec"][key] for key in ("kind", "factor") if key in description["question_spec"]}
    space = Space.from_dict(raw)
    arms(space, question)
    return space, question


def _observations(space, observations, identity, allowed_ids):
    if identity not in ("seed", "replicate"):
        raise ValueError("identity must be seed or replicate")
    cells, duplicates = {}, 0
    factors = {factor.name for factor in space.factors}
    allowed_ids = set(allowed_ids) if allowed_ids is not None else None
    for row in observations:
        if not isinstance(row.get("config"), dict) or set(row["config"]) != factors:
            raise ValueError("observations require a complete configuration")
        index = row.get(identity)
        if type(index) is not int or index < (1 if identity == "replicate" else 0):
            raise ValueError(f"observation requires a valid {identity} identity")
        if allowed_ids is not None and index not in allowed_ids:
            raise ValueError("observation identity is outside the visible partition")
        value = float(row["metrics"][space.metric])
        if not math.isfinite(value):
            raise ValueError("observations require a finite primary metric")
        key = config_key(space.normalize(row["config"]))
        cell = cells.setdefault(key, {})
        if index in cell:
            if cell[index] != value:
                raise ValueError("conflicting observations for one configuration and identity")
            duplicates += 1
        cell[index] = value
    return cells, duplicates


def _contrast(space, comparison, cells, fixed=None):
    factor, a, b = comparison
    contexts, pairs = [], []
    unmatched = 0
    for context in grid_over(space, [factor], fixed):
        va = cells.get(config_key({**context, factor: a}), {})
        vb = cells.get(config_key({**context, factor: b}), {})
        ids = sorted(va.keys() & vb.keys())
        unmatched += len(va.keys() ^ vb.keys())
        differences = [(va[index] - vb[index]) * sign(space) for index in ids]
        delta = float(np.mean(differences)) if differences else None
        if delta is not None and not math.isfinite(delta):
            raise ValueError("paired contrasts must be finite")
        contexts.append({"context": context, "paired_ids": ids, "n_pairs": len(ids), "observed_delta": delta,
                         "contribution": (delta if delta is not None else 0.0)})
        pairs.append(({index: va[index] * sign(space) for index in ids},
                      {index: vb[index] * sign(space) for index in ids}))
    # Every declared context retains weight 1/C. Unobserved context effects use a fixed neutral prior.
    delta = float(np.mean([row["contribution"] for row in contexts]))
    if not math.isfinite(delta):
        raise ValueError("target contrast must be finite")
    covered = sum(row["n_pairs"] > 0 for row in contexts)
    identity_sets = [set(a) for a, _ in pairs]
    balanced = covered == len(contexts) and all(ids == identity_sets[0] for ids in identity_sets)
    if balanced and len(identity_sets[0]) >= 2:
        mean, se = average_contrast(pairs)
        lo, hi = confidence_interval(mean, se, len(identity_sets[0]))
        uncertainty = {"status": "balanced_block_diagnostic", "n_blocks": len(identity_sets[0]), "delta": mean,
                       "se": se, "interval": [lo, hi], "interval_label": interval_label(lo, hi, space.min_effect)}
    else:
        uncertainty = {"status": "unavailable", "reason": "Requires the same two or more paired identities in every target context."}
    return {"answer": point_label(delta, space.min_effect), "delta": delta,
            "status": "no_pairs_fallback" if not covered else "complete_coverage" if covered == len(contexts) else "partial_coverage",
            "contexts_observed": covered, "contexts_total": len(contexts), "coverage": covered / len(contexts),
            "paired_observations": sum(row["n_pairs"] for row in contexts), "unmatched_observations": unmatched,
            "contexts": contexts, "uncertainty": uncertainty}


def estimate_answer(space, question, observations, *, identity="replicate", allowed_ids=None):
    """Forced point prediction plus coverage diagnostics, never a scientific equivalence attestation."""
    comparison = arms(space, question)
    cells, duplicates = _observations(space, observations, identity, allowed_ids)
    if question["kind"] == "where":
        factor = space.factor(question["factor"])
        contrasts = {(level if isinstance(level, str) else canon(level)):
                     _contrast(space, comparison, cells, fixed={factor.name: level}) for level in factor.levels}
        answer = {key: value["answer"] for key, value in contrasts.items()}
    else:
        contrasts = {"overall": _contrast(space, comparison, cells)}
        answer = contrasts["overall"]["answer"]
        if question["kind"] == "component":
            answer = COMPONENT_LABELS[answer]
    return {"estimator_id": ESTIMATOR_ID, "answer": answer, "identity": identity,
            "duplicates_ignored": duplicates, "contrasts": contrasts,
            "missing_context_rule": "Fixed zero effect for unobserved contexts; divide by the full target context count.",
            "limits": ["A neutral missing-context prior can bias answers toward no_meaningful_difference; it is not evidence of equivalence.",
                       "Point labels are forced predictions, not visible-data significance decisions.",
                       "Block intervals are descriptive balanced-design diagnostics, not validated sequential intervals or hidden-seed rewards."]}
