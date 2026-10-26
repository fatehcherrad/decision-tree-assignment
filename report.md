# Decision Tree algorithm

**Name:** Fateh Cherrad  
**GitHub:** https://github.com/fatehcherrad/decision-tree-assignment  
**PyPI:** https://pypi.org/project/fateh-simple-decision-tree/

## What the project does

This project implements a classification decision tree from scratch in Python. It uses only the standard library. The main class has three methods: `fit`, `predict` and `display`.

The example is the Play Tennis table used in class. There are 14 rows, with 9 Yes labels and 5 No labels. Outlook, Temperature, Humidity and Wind are the input features. Day is not used because it is a row identifier.

## How the algorithm works

At each node, the algorithm calculates entropy:

`H(S) = -sum(p * log2(p))`

It then tries the possible splits and calculates their information gain:

`Gain = H(parent) - sum((child_size / parent_size) * H(child))`

The split with the largest positive gain is selected. Categorical features produce one branch per value. Numeric features use a threshold between consecutive distinct values, producing two branches. The same process is repeated for the child nodes.

Training stops when all labels agree, the maximum depth is reached, no categorical features remain (and no numeric features are available), or no split improves the entropy. A leaf predicts the most common label at that node. If a prediction contains an unknown category, the current node's majority label is used.

## Tree size and results

The `max_depth` parameter limits tree size by restricting its depth. The root has depth 0. A larger value allows more questions, but can also lead to overfitting. It is a depth limit rather than an exact node-count limit.

These results were obtained by running `python example.py`:

| max_depth | Correct training predictions |
|---|---|
| 0 | 9/14 |
| 1 | 10/14 |
| 2 | 14/14 |

The initial entropy is about 0.9403. Outlook has the largest initial gain, about 0.2467, so it becomes the root. The resulting rules are:

- Overcast: Yes.
- Sunny: High humidity means No; Normal humidity means Yes.
- Rain: Weak wind means Yes; Strong wind means No.

The 14/14 result is on the training data. It does not show that the model will be perfect on new data.

## Checks and bonuses

Nine automated tests check entropy, information gain, the expected tennis tree, the depth limit, numeric predictions on unseen values, unknown categories, constant features, display output and invalid inputs. Run them with `python -m unittest discover -s tests -v`.

The display bonus is implemented as an indented text tree. The code is also packaged with `pyproject.toml` under the distribution name `fateh-simple-decision-tree`. The release workflow runs the tests, builds the package and publishes through PyPI Trusted Publishing, without storing an API token.

Version `0.1.0` was published to PyPI on 26 October 2026. Installation from the public index was checked in a clean environment, including entropy, prediction and display. Both bonuses are implemented.

```bash
python -m pip install fateh-simple-decision-tree==0.1.0
```

## Limitations

This version is intended for small datasets. It does not include post-pruning, and checking all numeric thresholds can be slow. A feature combination with no positive gain at the first split, such as XOR, is not learned by this stopping rule.

