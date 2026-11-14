"""Small, inspectable constraint engine for offline Train Tracks traces.

Rows and columns are zero-based internally. Masks use the tile order in TILES.
Every deduction removes only candidates ruled out by a visible local constraint.
"""

from __future__ import annotations

from dataclasses import dataclass

TILES = ("XX", "NS", "EW", "NE", "NW", "SE", "SW")
BIT = {name: 1 << index for index, name in enumerate(TILES)}
ALL = (1 << len(TILES)) - 1
TRACK = ALL ^ BIT["XX"]
DIRECTIONS = {"N": (-1, 0, "S"), "E": (0, 1, "W"),
              "S": (1, 0, "N"), "W": (0, -1, "E")}


def choices(mask: int) -> list[str]:
    return [name for name in TILES if mask & BIT[name]]


def one(mask: int) -> str | None:
    return TILES[mask.bit_length() - 1] if mask and mask.bit_count() == 1 else None


def clue_from_record(record: dict) -> dict:
    puzzle = record["puzzle"]
    height = len(puzzle["rows"])
    givens = [(height - 1 - row, column, shape)
              for shape, row, column, *_ in puzzle["pieces"]]
    return {"rows": puzzle["rows"], "columns": puzzle["columns"],
            "given_shapes": sorted(givens)}


def initial_domains(clue: dict) -> list[list[int]]:
    height, width = len(clue["rows"]), len(clue["columns"])
    given = {(r, c): shape for r, c, shape in clue["given_shapes"]}
    board = []
    for r in range(height):
        line = []
        for c in range(width):
            if (r, c) in given:
                line.append(BIT[given[r, c]])
                continue
            allowed = 0
            for name in TILES:
                if all(0 <= r + DIRECTIONS[d][0] < height
                       and 0 <= c + DIRECTIONS[d][1] < width for d in name if d != "X"):
                    allowed |= BIT[name]
            line.append(allowed)
        board.append(line)
    return board


def snapshot(board: list[list[int]]) -> list[list[int]]:
    return [line.copy() for line in board]


@dataclass(frozen=True)
class Step:
    kind: str  # deduce or contradiction
    rule: str
    anchor: dict
    cell: tuple[int, int] | None = None
    mask: int | None = None

    def action(self) -> dict:
        result = {"type": self.kind, "rule": self.rule, "anchor": self.anchor}
        if self.cell is not None:
            result["cell"] = {"row": self.cell[0] + 1,
                              "column": self.cell[1] + 1}
        if self.mask is not None:
            result["candidates"] = choices(self.mask)
        return result


def _line_step(board, cells, clue, kind, index):
    fixed = sum(not (board[r][c] & BIT["XX"]) for r, c in cells)
    possible = sum(bool(board[r][c] & TRACK) for r, c in cells)
    anchor = {kind: index + 1, "clue": clue}
    if fixed > clue or possible < clue:
        return Step("contradiction", "clue_bounds", anchor)
    for r, c in cells:
        old = board[r][c]
        if fixed == clue and old & BIT["XX"] and old & TRACK:
            return Step("deduce", "clue_satisfied", anchor,
                        (r, c), BIT["XX"])
        if possible == clue and old & BIT["XX"] and old & TRACK:
            return Step("deduce", "all_remaining_cells_required", anchor,
                        (r, c), old & TRACK)
    return None


def _port_step(board, r, c):
    height, width = len(board), len(board[0])
    old = board[r][c]
    supported = 0
    for shape in choices(old):
        if shape == "XX":
            ports = ""
        else:
            ports = shape
        valid = True
        for direction, (dr, dc, opposite) in DIRECTIONS.items():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue  # The boundary and endpoint rule is set initially.
            if not any((direction in ports) == (opposite in neighbor)
                       for neighbor in choices(board[nr][nc])):
                valid = False
                break
        if valid:
            supported |= BIT[shape]
    if supported == old:
        return None
    anchor = {"row": r + 1, "column": c + 1}
    return (Step("deduce", "port_reciprocity", anchor, (r, c), supported)
            if supported else Step("contradiction", "port_reciprocity", anchor,
                                   (r, c)))


def next_step(board: list[list[int]], clue: dict) -> Step | None:
    height, width = len(board), len(board[0])
    for r in range(height):
        event = _line_step(board, [(r, c) for c in range(width)],
                           clue["rows"][r], "row", r)
        if event:
            return event
    for c in range(width):
        event = _line_step(board, [(r, c) for r in range(height)],
                           clue["columns"][c], "column", c)
        if event:
            return event
    for r in range(height):
        for c in range(width):
            event = _port_step(board, r, c)
            if event:
                return event
    if all(one(mask) is not None for line in board for mask in line):
        error = completed_error(board, clue)
        if error:
            return Step("contradiction", error, {"board": "complete"})
    return None


def completed_error(board: list[list[int]], clue: dict) -> str | None:
    height, width = len(board), len(board[0])
    grid = [[one(mask) for mask in line] for line in board]
    if any(tile is None for line in grid for tile in line):
        return "incomplete"
    if any(sum(tile != "XX" for tile in grid[r]) != clue["rows"][r]
           for r in range(height)) or any(
               sum(grid[r][c] != "XX" for r in range(height)) != clue["columns"][c]
               for c in range(width)):
        return "clue_mismatch"
    occupied = {(r, c) for r in range(height) for c in range(width)
                if grid[r][c] != "XX"}
    given = {(r, c): shape for r, c, shape in clue["given_shapes"]}
    if any(grid[r][c] != shape for (r, c), shape in given.items()):
        return "changed_given"
    external = []
    for r, c in occupied:
        for direction in grid[r][c]:
            dr, dc, opposite = DIRECTIONS[direction]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < height and 0 <= nc < width):
                external.append((r, c, direction))
            elif opposite not in grid[nr][nc]:
                return "port_reciprocity"
    expected_external = {(r, c, direction) for (r, c), shape in given.items()
                         for direction in shape if not (
                             0 <= r + DIRECTIONS[direction][0] < height and
                             0 <= c + DIRECTIONS[direction][1] < width)}
    if set(external) != expected_external or len(external) != 2:
        return "endpoints"
    if not occupied:
        return "empty_path"
    found, todo = set(), [next(iter(occupied))]
    while todo:
        cell = todo.pop()
        if cell in found:
            continue
        found.add(cell)
        r, c = cell
        for direction in grid[r][c]:
            dr, dc, _ = DIRECTIONS[direction]
            neighbor = r + dr, c + dc
            if neighbor in occupied and neighbor not in found:
                todo.append(neighbor)
    return None if found == occupied else "disconnected_path"


def apply_deduction(board, step: Step) -> None:
    if step.kind != "deduce" or step.cell is None or step.mask is None:
        raise ValueError("Expected a deduction")
    r, c = step.cell
    if step.mask == 0 or step.mask == board[r][c] or step.mask & board[r][c] != step.mask:
        raise ValueError("Deduction must remove at least one candidate")
    board[r][c] = step.mask


def branch_cell(board: list[list[int]]) -> tuple[int, int] | None:
    candidates = [(mask.bit_count(), r, c) for r, line in enumerate(board)
                  for c, mask in enumerate(line) if mask.bit_count() > 1]
    return min(candidates)[1:] if candidates else None
