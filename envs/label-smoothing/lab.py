import numpy as np
import torch
from sklearn.datasets import load_digits

_X, _Y = load_digits(return_X_y=True)
_X = (_X / 16.0).astype(np.float32)


def run(config, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(_X))
    test_idx, pool = order[:500], order[500:]
    train_idx = pool[: config["n_train"]]

    y_train = _Y[train_idx].copy()
    flip = rng.random(len(y_train)) < config["label_noise"]
    y_train[flip] = rng.integers(0, 10, flip.sum())

    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(64, config["hidden"]), torch.nn.ReLU(), torch.nn.Linear(config["hidden"], 10)
    )
    smoothing = 0.2 if config["method"] == "label_smoothing" else 0.0
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=smoothing)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    x, y = torch.from_numpy(_X[train_idx]), torch.from_numpy(y_train).long()
    for _ in range(150):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits = model(torch.from_numpy(_X[test_idx]))
        probs = logits.softmax(-1)
        y_test = torch.from_numpy(_Y[test_idx]).long()
        acc = (logits.argmax(-1) == y_test).float().mean().item()
        nll = torch.nn.functional.cross_entropy(logits, y_test).item()
        conf = probs.max(-1).values.mean().item()
    return {"test_acc": acc, "test_nll": nll, "mean_confidence": conf, "train_loss": loss.item()}
