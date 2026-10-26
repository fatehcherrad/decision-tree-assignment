import unittest
from example import load_tennis
from simple_tree import DecisionTree, entropy, information_gain


class TreeTests(unittest.TestCase):
    def test_entropy(self):
        self.assertEqual(entropy([]), 0)
        self.assertEqual(entropy(["Yes"] * 4), 0)
        self.assertAlmostEqual(entropy(["Yes", "No"]), 1)
        self.assertAlmostEqual(entropy(["Yes"] * 9 + ["No"] * 5), 0.9402859587)

    def test_outlook_gain(self):
        X, y = load_tennis()
        groups = [[label for row, label in zip(X, y) if row["Outlook"] == value]
                  for value in ("Sunny", "Overcast", "Rain")]
        self.assertAlmostEqual(information_gain(y, groups), 0.2467498198)

    def test_tennis_structure_and_predictions(self):
        X, y = load_tennis()
        model = DecisionTree(max_depth=2).fit(X, y)
        self.assertEqual(model.root.feature, "Outlook")
        self.assertEqual(model.root.children["Sunny"].feature, "Humidity")
        self.assertEqual(model.root.children["Rain"].feature, "Wind")
        self.assertEqual(model.predict(X), y)

    def test_depth_changes_tree_size(self):
        X, y = load_tennis()
        zero = DecisionTree(0).fit(X, y)
        self.assertEqual(zero.predict(X), ["Yes"] * 14)
        one = DecisionTree(1).fit(X, y)
        self.assertTrue(all(not n.children for n in one.root.children.values()))
        self.assertEqual(sum(a == b for a, b in zip(one.predict(X), y)), 10)

    def test_numeric_threshold_on_unseen_values(self):
        model = DecisionTree(1).fit([{"x": n} for n in (1, 2, 8, 9)], ["Low", "Low", "High", "High"])
        self.assertEqual(model.root.threshold, 5)
        self.assertEqual(model.predict([{"x": 3}, {"x": 7}]), ["Low", "High"])

    def test_unseen_category_and_missing_feature(self):
        X, y = load_tennis()
        model = DecisionTree(2).fit(X, y)
        self.assertEqual(model.predict([{"Outlook": "Snow"}, {}]), ["Yes", "Yes"])
        self.assertEqual(model.predict([{"Outlook": "Sunny", "Humidity": "Unknown"}]), ["No"])

    def test_constant_feature(self):
        model = DecisionTree().fit([{"x": "a"}] * 3, ["No", "Yes", "Yes"])
        self.assertFalse(model.root.children)
        self.assertEqual(model.predict([{"x": "a"}]), ["Yes"])

    def test_display(self):
        X, y = load_tennis()
        output = DecisionTree(2).fit(X, y).display()
        self.assertIn("Outlook?", output)
        self.assertIn("Predict No", output)

    def test_invalid_inputs(self):
        for depth in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                DecisionTree(depth)
        for X, y in [([], []), ([{"x": 1}], []), ([{"x": None}], ["a"]),
                     ([{"x": float("nan")}], ["a"]), ([{"x": 1}, {"z": 2}], ["a", "b"])]:
            with self.assertRaises(ValueError):
                DecisionTree().fit(X, y)
        with self.assertRaises(ValueError):
            DecisionTree().predict([{}])


if __name__ == "__main__":
    unittest.main()
