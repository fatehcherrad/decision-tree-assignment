# Decision tree from scratch

A small Python classification tree for the decision-tree assignment. The splitting algorithm uses entropy and information gain. It does not use scikit-learn, NumPy, or a prebuilt tree implementation.

## Run the example

Python 3.10 or newer is required. From the project folder:

```bash
python example.py
python -m unittest discover -s tests -v
```

The example uses the 14-row Play Tennis table from class. The `Day` column is excluded because it is only an identifier. It prints the results for different depth limits and displays the tree.

## Use the class

```python
from simple_tree import DecisionTree

X = [{"temperature": 10}, {"temperature": 15}, {"temperature": 25}, {"temperature": 30}]
y = ["Cold", "Cold", "Warm", "Warm"]

tree = DecisionTree(max_depth=2)
tree.fit(X, y)
print(tree.predict([{"temperature": 20}]))
print(tree.display())
```

Input rows are dictionaries. Class labels and categorical values must be hashable, such as strings or integers. Numeric values are split at candidate thresholds; strings and booleans are treated as categories. Missing training values and non-finite numeric values are rejected. At prediction time, a missing feature or an unseen category uses the current node's majority class.

`max_depth` controls the longest path from the root to a leaf. Root depth is zero. This limits tree depth, not the exact number of nodes: a categorical split can have several children. Equal class counts are resolved by first appearance; equal gains keep the first candidate.

## Files

- `simple_tree/tree.py`: entropy, information gain, training, prediction and text display.
- `example.py` and `data/tennis.csv`: the classroom example.
- `tests/test_tree.py`: nine tests.
- `report.md`: short assignment report.
- `pyproject.toml`: installable package metadata.

## Package

Published releases are available on [PyPI](https://pypi.org/project/fateh-simple-decision-tree/).

Install locally with `python -m pip install .`. Distribution name: `fateh-simple-decision-tree`; import name: `simple_tree`.

Build and check a distribution:

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

To publish, the owner needs a PyPI account and permission to create this package:

```bash
python -m twine upload dist/*
```

The manually triggered GitHub Actions workflow `.github/workflows/publish.yml` builds, tests and publishes releases using PyPI Trusted Publishing.

Package page: https://pypi.org/project/fateh-simple-decision-tree/

Install a published release with `python -m pip install fateh-simple-decision-tree`.

## Limits

This is an educational implementation for small datasets. It checks candidate numeric thresholds directly, which is slow for large datasets. There is no pruning after training, missing-value imputation, or probability prediction. The tree stops if no split has positive gain; this can miss patterns such as XOR. The example's training accuracy is not an estimate of performance on unseen data.

