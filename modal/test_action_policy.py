"""Local rule-checker tests: python -m unittest discover -s modal -p 'test_*.py'."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from action_policy import development_ids, validate_action


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "data/train.jsonl"
        with path.open(encoding="utf-8") as handle:
            cls.rows = [json.loads(line) for line in handle]

    def test_positive_trace_actions_are_rule_valid(self):
        for row in self.rows:
            if not row["expert_policy_label"]:
                continue
            state = row["state"]
            clue = {"rows": state["row_clues"], "columns": state["column_clues"],
                    "given_shapes": [(g["row"] - 1, g["column"] - 1, g["value"])
                                     for g in state["givens"]]}
            ok, reason, _ = validate_action(state["domains"], clue, row["action"],
                                            state["branch_stack"],
                                            row["action"]["type"] == "undo")
            self.assertTrue(ok, (row["puzzle_id"], row["step_index"], reason))

    def test_false_deduction_is_rejected(self):
        row = self.rows[0]
        state = row["state"]
        clue = {"rows": state["row_clues"], "columns": state["column_clues"],
                "given_shapes": [(g["row"] - 1, g["column"] - 1, g["value"])
                                 for g in state["givens"]]}
        wrong = dict(row["action"], candidates=["NS"])
        ok, _, _ = validate_action(state["domains"], clue, wrong, [], False)
        self.assertFalse(ok)

    def test_split_is_puzzle_disjoint(self):
        ids = {r["puzzle_id"] for r in self.rows}
        dev = development_ids(ids)
        self.assertEqual(len(dev), 32)
        self.assertFalse(dev & (ids - dev))


if __name__ == "__main__":
    unittest.main()
