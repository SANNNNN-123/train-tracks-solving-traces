"""Shared text policy format and rule checker for interactive trace rollouts."""

import hashlib
import json

from trace_logic import (BIT, TRACK, TILES, Step, _line_step, _port_step,
                         branch_cell, choices, completed_error, one)

SYSTEM = ("Solve Train Tracks one action at a time. Choose only from the "
          "provided legal-action menu. Return exactly one JSON object with the "
          "selected id, for example {\"choice\":\"a00\"}. Do not invent an "
          "action or alter a listed action.")


def development_ids(puzzle_ids, count=32):
    ids = sorted(set(puzzle_ids), key=lambda x: hashlib.sha256(x.encode()).hexdigest())
    if len(ids) <= count:
        raise ValueError("Not enough puzzles for a disjoint train/development split")
    return set(ids[:count])


def _action_key(action):
    return json.dumps(action, sort_keys=True, separators=(",", ":"))


def legal_actions(state, pending=False):
    """Enumerate rule-supported actions; offer branches only when propagation stalls."""
    board = state["domains"]
    clue = {"rows": state["row_clues"], "columns": state["column_clues"],
            "given_shapes": [(g["row"] - 1, g["column"] - 1, g["value"])
                             for g in state["givens"]]}
    if pending:
        stack = state["branch_stack"]
        return ([{"type": "undo", "to_step": stack[-1]["to_step"]}]
                if stack else [])

    height, width = len(board), len(board[0])
    stack = state["branch_stack"]
    actions = []
    for row in range(height):
        cells = [(row, c) for c in range(width)]
        fixed = sum(not (board[r][c] & BIT["XX"]) for r, c in cells)
        possible = sum(bool(board[r][c] & TRACK) for r, c in cells)
        number = clue["rows"][row]
        anchor = {"row": row + 1, "clue": number}
        if stack and (fixed > number or possible < number):
            actions.append(Step("contradiction", "clue_bounds", anchor).action())
        for c in range(width):
            cell = (row, c)
            for rule in ("clue_satisfied", "all_remaining_cells_required"):
                step = _rule_step(board, clue, rule, anchor, cell)
                if step:
                    actions.append(step.action())
    for column in range(width):
        cells = [(r, column) for r in range(height)]
        fixed = sum(not (board[r][c] & BIT["XX"]) for r, c in cells)
        possible = sum(bool(board[r][c] & TRACK) for r, c in cells)
        number = clue["columns"][column]
        anchor = {"column": column + 1, "clue": number}
        if stack and (fixed > number or possible < number):
            actions.append(Step("contradiction", "clue_bounds", anchor).action())
        for r in range(height):
            cell = (r, column)
            for rule in ("clue_satisfied", "all_remaining_cells_required"):
                step = _rule_step(board, clue, rule, anchor, cell)
                if step:
                    actions.append(step.action())
    for r in range(height):
        for c in range(width):
            step = _port_step(board, r, c)
            if step and (step.kind == "deduce" or stack):
                actions.append(step.action())

    error = completed_error(board, clue)
    if error is None:
        actions.append({"type": "finish"})
    elif stack and error != "incomplete":
        actions.append(Step("contradiction", error, {"board": "complete"}).action())

    # A branch is a fallback after no visible deduction remains.
    if not actions:
        cell = branch_cell(board)
        if cell:
            r, c = cell
            for value in choices(board[r][c]):
                actions.append({"type": "branch",
                                "cell": {"row": r + 1, "column": c + 1},
                                "value": value})

    unique = {_action_key(action): action for action in actions}
    return [unique[key] for key in sorted(unique)]


def choice_id(action, actions):
    key = _action_key(action)
    for index, candidate in enumerate(actions):
        if _action_key(candidate) == key:
            return f"a{index:03d}"
    raise ValueError(f"Trace action is absent from its legal menu: {action}")


def prompt(state, actions):
    readable = dict(state)
    readable["domains"] = [[choices(mask) for mask in row]
                           for row in state["domains"]]
    readable["legal_actions"] = [
        {"id": f"a{index:03d}", "action": action}
        for index, action in enumerate(actions)]
    return ("Board state (each domains cell lists its remaining tiles; "
            "XX means empty):\n"
            + json.dumps(readable, separators=(",", ":"), ensure_ascii=False)
            + "\nChoose one listed legal action. JSON choice id only.")


def completion(action_id):
    return json.dumps({"choice": action_id}, separators=(",", ":"))


def parse_action(raw, actions):
    try:
        result = json.loads(raw.strip())
    except (ValueError, TypeError):
        return None
    if not isinstance(result, dict) or set(result) != {"choice"}:
        return None
    choice = result["choice"]
    if not isinstance(choice, str) or not choice.startswith("a"):
        return None
    try:
        index = int(choice[1:])
    except ValueError:
        return None
    return actions[index] if 0 <= index < len(actions) and choice == f"a{index:03d}" else None


def _cell(action, board):
    cell = action.get("cell")
    if not isinstance(cell, dict):
        return None
    r, c = cell.get("row"), cell.get("column")
    if type(r) is not int or type(c) is not int or not (1 <= r <= len(board)) or not (1 <= c <= len(board[0])):
        return None
    return r - 1, c - 1


def _rule_step(board, clue, rule, anchor, cell=None):
    if not isinstance(rule, str):
        return None
    h, w = len(board), len(board[0])
    if rule in {"clue_satisfied", "all_remaining_cells_required", "clue_bounds"}:
        if not isinstance(anchor, dict):
            return None
        if "row" in anchor and type(anchor["row"]) is int and 1 <= anchor["row"] <= h:
            i = anchor["row"] - 1
            cells = [(i, c) for c in range(w)]
            kind, number = "row", clue["rows"][i]
        elif "column" in anchor and type(anchor["column"]) is int and 1 <= anchor["column"] <= w:
            i = anchor["column"] - 1
            cells = [(r, i) for r in range(h)]
            kind, number = "column", clue["columns"][i]
        else:
            return None
        if anchor != {kind: i + 1, "clue": number}:
            return None
        fixed = sum(not (board[r][c] & BIT["XX"]) for r, c in cells)
        possible = sum(bool(board[r][c] & TRACK) for r, c in cells)
        if rule == "clue_bounds":
            return Step("contradiction", rule, anchor) if fixed > number or possible < number else None
        if cell not in cells:
            return None
        r, c = cell
        old = board[r][c]
        if not (old & BIT["XX"] and old & TRACK):
            return None
        if rule == "clue_satisfied" and fixed == number:
            return Step("deduce", rule, anchor, cell, BIT["XX"])
        if rule == "all_remaining_cells_required" and possible == number:
            return Step("deduce", rule, anchor, cell, old & TRACK)
    elif rule == "port_reciprocity" and isinstance(anchor, dict):
        if type(anchor.get("row")) is not int or type(anchor.get("column")) is not int:
            return None
        r, c = anchor["row"] - 1, anchor["column"] - 1
        if not (0 <= r < h and 0 <= c < w):
            return None
        if cell is not None and cell != (r, c):
            return None
        return _port_step(board, r, c)
    elif rule in {"disconnected_path", "endpoints", "empty_path", "clue_mismatch", "changed_given"}:
        if anchor == {"board": "complete"} and completed_error(board, clue) == rule:
            return Step("contradiction", rule, anchor)
    return None


def validate_action(board, clue, action, stack, pending):
    """Return (valid, reason, canonical action). No solution is consulted."""
    if not isinstance(action, dict):
        return False, "malformed", None
    kind = action.get("type")
    if not isinstance(kind, str):
        return False, "malformed", None
    if pending:
        if kind != "undo" or not stack or action.get("to_step") != stack[-1]["to_step"]:
            return False, "expected_undo", None
        return True, "ok", {"type": "undo", "to_step": stack[-1]["to_step"]}
    if kind in {"deduce", "contradiction"}:
        cell = _cell(action, board) if "cell" in action else None
        if "cell" in action and cell is None:
            return False, "invalid_cell", None
        step = _rule_step(board, clue, action.get("rule"), action.get("anchor"), cell)
        if step is None or step.kind != kind:
            return False, "unsupported_rule", None
        if kind == "deduce":
            if cell is None or step.cell != cell or action.get("candidates") != choices(step.mask):
                return False, "wrong_candidates", None
        elif step.cell is not None and step.cell != cell:
            return False, "wrong_contradiction_cell", None
        elif step.cell is None and cell is not None:
            return False, "unexpected_cell", None
        return True, "ok", step.action()
    if kind == "branch":
        cell = _cell(action, board)
        value = action.get("value")
        if cell is None or not isinstance(value, str) or value not in BIT or board[cell[0]][cell[1]].bit_count() < 2:
            return False, "invalid_branch", None
        if not board[cell[0]][cell[1]] & BIT[value]:
            return False, "branch_outside_domain", None
        if completed_error(board, clue) not in {"incomplete", None}:
            return False, "contradictory_board", None
        return True, "ok", {"type": "branch", "cell": action["cell"], "value": value}
    if kind == "finish":
        return (True, "ok", {"type": "finish"}) if completed_error(board, clue) is None else (False, "unfinished", None)
    return False, "unexpected_action", None


def split_examples(rows, max_deductions_per_puzzle=48):
    """Keep every start and rare positive action; balance sampled deductions by rule."""
    from collections import defaultdict
    grouped = defaultdict(list)
    for row in rows:
        if row.get("expert_policy_label"):
            grouped[row["puzzle_id"]].append(row)
    selected = []
    for puzzle_id, items in grouped.items():
        deductions = [r for r in items if r["action"]["type"] == "deduce"]
        rare = [r for r in items if r["action"]["type"] != "deduce"]
        starts = [r for r in items if r["step_index"] == 0]
        selected.extend(starts)

        start_steps = {r["step_index"] for r in starts}
        by_rule = defaultdict(list)
        for row in deductions:
            if row["step_index"] not in start_steps:
                by_rule[row["action"].get("rule", "unknown")].append(row)
        for rule, rule_rows in by_rule.items():
            rule_rows.sort(key=lambda r: hashlib.sha256(
                f"{puzzle_id}:{r['step_index']}:42".encode()).hexdigest())

        start_deductions = sum(r["action"]["type"] == "deduce" for r in starts)
        remaining = max(0, max_deductions_per_puzzle - start_deductions)
        rules = sorted(by_rule)
        while remaining and rules:
            active = []
            for rule in rules:
                if by_rule[rule]:
                    selected.append(by_rule[rule].pop())
                    remaining -= 1
                    if by_rule[rule]:
                        active.append(rule)
                    if remaining == 0:
                        break
            rules = active
        selected.extend(r for r in rare if r["step_index"] not in start_steps)
    selected.sort(key=lambda r: hashlib.sha256(
        f"{r['puzzle_id']}:{r['step_index']}:shuffle".encode()).hexdigest())
    return selected
