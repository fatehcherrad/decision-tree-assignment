"""A small classification tree using entropy and information gain."""

from collections import Counter
from dataclasses import dataclass, field
from math import isfinite, log2
from numbers import Real


def entropy(labels):
    if not labels:
        return 0.0
    counts = Counter(labels)
    return -sum((n / len(labels)) * log2(n / len(labels)) for n in counts.values())


def information_gain(parent, groups):
    if not parent:
        return 0.0
    remaining = sum(len(group) / len(parent) * entropy(group) for group in groups)
    return entropy(parent) - remaining


@dataclass
class Node:
    prediction: object
    samples: int
    feature: str | None = None
    threshold: float | None = None
    children: dict = field(default_factory=dict)


class DecisionTree:
    """ID3-style categorical splits and binary numeric splits.

    X is a list of dictionaries; y contains the class labels.
    Root depth is zero. max_depth=0 makes a majority-class leaf.
    """

    def __init__(self, max_depth=3):
        if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0:
            raise ValueError("max_depth must be a non-negative integer")
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        if not X or len(X) != len(y):
            raise ValueError("X and y must be non-empty and have the same length")
        features = list(X[0])
        if any(set(row) != set(features) for row in X):
            raise ValueError("Every row must have the same features")
        self.numeric = {}
        for name in features:
            values = [row[name] for row in X]
            if any(v is None for v in values):
                raise ValueError("Missing values are not supported")
            numeric = [isinstance(v, Real) and not isinstance(v, bool) for v in values]
            if any(numeric) and not all(numeric):
                raise ValueError("A feature cannot mix numeric and categorical values")
            self.numeric[name] = all(numeric)
            if all(numeric) and any(not isfinite(v) for v in values):
                raise ValueError("Numeric features must be finite")
        self.root = self._build(X, y, features, 0)
        return self

    def _build(self, X, y, features, depth):
        # Counter breaks tied class counts by first appearance.
        majority = Counter(y).most_common(1)[0][0]
        node = Node(majority, len(y))
        if len(set(y)) == 1 or depth >= self.max_depth or not features:
            return node

        best_gain = 0.0
        best = None
        for name in features:
            if self.numeric[name]:
                values = sorted(set(row[name] for row in X))
                thresholds = [a / 2 + b / 2 for a, b in zip(values, values[1:])]
            else:
                thresholds = [None]
            for threshold in thresholds:
                groups = {}
                for i, row in enumerate(X):
                    key = row[name] if threshold is None else row[name] <= threshold
                    groups.setdefault(key, []).append(i)
                if len(groups) < 2:
                    continue
                gain = information_gain(y, [[y[i] for i in ids] for ids in groups.values()])
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best = name, threshold, groups

        # No useful split: keep the majority prediction.
        if best is None:
            return node
        name, threshold, groups = best
        node.feature, node.threshold = name, threshold
        remaining = features if self.numeric[name] else [f for f in features if f != name]
        for key, ids in groups.items():
            node.children[key] = self._build(
                [X[i] for i in ids], [y[i] for i in ids], remaining, depth + 1
            )
        return node

    def predict(self, X):
        if self.root is None:
            raise ValueError("Call fit before predict")
        return [self._predict_one(row) for row in X]

    def _predict_one(self, row):
        node = self.root
        while node.children:
            value = row.get(node.feature)
            if value is None:
                return node.prediction
            if node.threshold is not None:
                if not isinstance(value, Real) or not isfinite(value):
                    return node.prediction
                key = value <= node.threshold
            else:
                key = value
            if key not in node.children:
                return node.prediction
            node = node.children[key]
        return node.prediction

    def display(self):
        """Return a text tree, suitable for print(tree.display())."""
        if self.root is None:
            raise ValueError("Call fit before display")
        lines = []

        def visit(node, indent):
            if not node.children:
                lines.append(f"{indent}Predict {node.prediction} (n={node.samples})")
                return
            lines.append(f"{indent}{node.feature}? (n={node.samples})")
            for value, child in node.children.items():
                if node.threshold is None:
                    condition = f"= {value}"
                else:
                    condition = f"{'<=' if value else '>'} {node.threshold:g}"
                lines.append(f"{indent}  {condition}:")
                visit(child, indent + "    ")

        visit(self.root, "")
        return "\n".join(lines)
