import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

ROLES = ("treatment", "context", "component")
_MISSING = object()
MAX_CONFIGS = 96


def canon(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


_KEYS: dict = {}


def config_key(config: dict) -> str:
    # same json key as before, cached on the exact (name, type, value) items
    items = tuple(sorted((k, type(v), v) for k, v in config.items()))
    key = _KEYS.get(items)
    if key is None:
        key = _KEYS[items] = canon(config)
    return key


@dataclass(frozen=True)
class Factor:
    name: str
    levels: tuple
    role: str
    description: str = ""

    def __post_init__(self):
        object.__setattr__(self, "_index", {canon(level): level for level in self.levels})
        # keyed by type too, so 1, 1.0 and True stay distinct
        object.__setattr__(self, "_typed", {(type(level), level): level for level in self.levels})

    def match(self, value):
        try:
            exact = self._typed.get((type(value), value), _MISSING)
        except TypeError:
            exact = _MISSING
        if exact is not _MISSING:
            return exact
        exact = self._index.get(canon(value), _MISSING)
        if exact is not _MISSING:
            return exact
        for level in self.levels:
            if _loose_equal(level, value):
                return level
        raise ValueError(f"{self.name}={value!r} is not a level; pick one of {list(self.levels)}")


def _loose_equal(level, value) -> bool:
    if isinstance(level, bool) or isinstance(value, bool):
        return str(level).lower() == str(value).lower()
    if isinstance(level, (int, float)):
        try:
            return math.isclose(float(level), float(value), rel_tol=1e-9, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return str(level).strip().lower() == str(value).strip().lower()


@dataclass(frozen=True)
class Space:
    title: str
    factors: tuple[Factor, ...]
    metric: str
    goal: str
    min_effect: float
    treatment: str
    paper_level: object
    baseline_level: object
    metric_description: str = ""
    setting: str = ""
    default: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict) -> "Space":
        factors = []
        for f in raw["factors"]:
            levels = tuple(f["levels"])
            if len(levels) < 2:
                raise ValueError(f"factor {f['name']} needs at least 2 levels")
            if len({canon(v) for v in levels}) != len(levels):
                raise ValueError(f"factor {f['name']} has duplicate levels")
            role = f.get("role", "context")
            if role not in ROLES:
                raise ValueError(f"factor {f['name']} has unknown role {role}")
            factors.append(Factor(f["name"], levels, role, f.get("description", "")))
        names = [f.name for f in factors]
        if len(set(names)) != len(names):
            raise ValueError("factor names must be unique")

        metric = raw["metric"]
        goal = metric.get("goal", "max")
        if goal not in ("max", "min"):
            raise ValueError("metric goal must be max or min")
        min_effect = float(metric.get("min_effect", 0.0))
        if not min_effect > 0:
            raise ValueError("metric.min_effect must be positive")

        t = raw["treatment"]
        by_name = {f.name: f for f in factors}
        if t["factor"] not in by_name or by_name[t["factor"]].role != "treatment":
            raise ValueError("treatment.factor must name the factor with role treatment")
        tf = by_name[t["factor"]]
        paper, baseline = tf.match(t["paper"]), tf.match(t["baseline"])
        if canon(paper) == canon(baseline):
            raise ValueError("paper and baseline levels must differ")
        if sum(f.role == "treatment" for f in factors) != 1:
            raise ValueError("exactly one factor must have role treatment")

        default = {}
        for f in factors:
            given = raw.get("default", {}).get(f.name, f.levels[0])
            default[f.name] = f.match(given)

        space = cls(
            title=raw.get("title", ""),
            factors=tuple(factors),
            metric=metric["name"],
            goal=goal,
            min_effect=min_effect,
            treatment=tf.name,
            paper_level=paper,
            baseline_level=baseline,
            metric_description=metric.get("description", ""),
            setting=raw.get("setting", ""),
            default=default,
        )
        if space.size() > MAX_CONFIGS:
            raise ValueError(f"grid has {space.size()} configs; keep it at {MAX_CONFIGS} or fewer")
        return space

    @classmethod
    def load(cls, path: Path) -> "Space":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def factor(self, name: str) -> Factor:
        for f in self.factors:
            if f.name == name:
                return f
        raise KeyError(name)

    def size(self) -> int:
        return math.prod(len(f.levels) for f in self.factors)

    def configs(self) -> list[dict]:
        names = [f.name for f in self.factors]
        return [
            dict(zip(names, combo, strict=False))
            for combo in itertools.product(*(f.levels for f in self.factors))
        ]

    def normalize(self, config: dict) -> dict:
        known = {f.name for f in self.factors}
        unknown = set(config) - known
        if unknown:
            raise ValueError(f"unknown factors {sorted(unknown)}; factors are {sorted(known)}")
        return {
            f.name: f.match(config[f.name]) if f.name in config else self.default[f.name]
            for f in self.factors
        }

    def better(self, a: float, b: float) -> float:
        # positive means a beats b, whatever the goal
        return a - b if self.goal == "max" else b - a

    def context_factors(self) -> list[Factor]:
        return [f for f in self.factors if f.role == "context"]

    def component_factors(self) -> list[Factor]:
        return [f for f in self.factors if f.role == "component"]
