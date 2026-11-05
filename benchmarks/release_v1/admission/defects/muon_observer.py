"""Instrumentation appended only to disposable, source-bound Muon fixture copies."""

import hashlib

import numpy as np


def install_probe(namespace):
    original_run = namespace["run"]
    original_grads = namespace["_grads"]
    original_perturbed = namespace["_perturbed"]
    state = {}

    def fingerprint(name, arrays):
        digest = hashlib.sha256()
        for array in arrays:
            if hasattr(array, "detach"):
                array = array.detach().numpy()
            array = np.ascontiguousarray(array)
            digest.update(str((array.shape, array.dtype.str)).encode())
            digest.update(array.tobytes())
        # Worker metrics must be finite numbers; eight 32-bit limbs preserve all SHA-256 bits.
        value = digest.digest()
        for i in range(8):
            state["metrics"][f"fixture.{name}.{i}"] = int.from_bytes(value[4*i:4*i+4], "big")

    def capture(tr, te, batches, parameters, xtr, ytr, xte, yte):
        fingerprint("train", [tr, xtr, ytr])
        fingerprint("test", [te, xte, yte])
        fingerprint("batches", [batches])
        fingerprint("initial_parameters", parameters)
        state["metrics"]["fixture.train_test_overlap"] = len(set(tr.tolist()) & set(te.tolist()))

    def grads(*args):
        state["gradient_calls"] += 1
        return original_grads(*args)

    def perturbed(parameters, directions, radius):
        result = original_perturbed(parameters, directions, radius)
        norm = sum(float(((a - b) ** 2).sum()) for a, b in zip(result, parameters, strict=True)) ** 0.5
        state["perturbation_norms"].append(norm)
        return result

    def run(config, seed):
        state.clear()
        state.update(metrics={}, gradient_calls=0, perturbation_norms=[])
        result = original_run(config, seed)
        # The final perturbation/gradient belongs to the train-sharpness diagnostic.
        training = state["perturbation_norms"][:-1]
        return {**result, **state["metrics"],
                "fixture.gradient_calls": state["gradient_calls"],
                "fixture.training_perturbation_calls": len(training),
                "fixture.first_training_perturbation_norm": training[0] if training else 0.0,
                "fixture.later_max_perturbation_norm": max(training[1:], default=0.0)}

    namespace.update(run=run, _grads=grads, _perturbed=perturbed, _fixture_capture=capture)
