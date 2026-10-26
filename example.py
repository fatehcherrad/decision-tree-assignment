import csv
from pathlib import Path
from simple_tree import DecisionTree


def load_tennis():
    path = Path(__file__).parent / "data" / "tennis.csv"
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    X = [{key: value for key, value in row.items() if key not in ("Day", "Play")} for row in rows]
    y = [row["Play"] for row in rows]
    return X, y


if __name__ == "__main__":
    X, y = load_tennis()
    for depth in (0, 1, 2):
        tree = DecisionTree(max_depth=depth).fit(X, y)
        correct = sum(a == b for a, b in zip(tree.predict(X), y))
        print(f"max_depth={depth}: {correct}/{len(y)} correct on training data")
    print("\nTree with max_depth=2:")
    print(tree.display())
    day = {"Outlook": "Rain", "Temperature": "Mild", "Humidity": "High", "Wind": "Strong"}
    print("\nNew prediction:", tree.predict([day])[0])
