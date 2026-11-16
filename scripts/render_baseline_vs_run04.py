"""Render the base-model vs fine-tuned comparison page.

Visual language and board renderer come from `solving-traces-V2.html`
(same palette, clue rendering, SVG tile art, step slider, puzzle select), but
the page shows two policies on the same puzzle: zero-shot Qwen2.5-1.5B on the
left, the run-04 adapter on the right.

    python scripts/render_baseline_vs_run04.py

Outputs `comparison.html` next to the eval reports, self-contained: the six
track tiles are inlined as data URIs, so the file can be moved or emailed alone.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle"))

from trace_logic import BIT, clue_from_record, initial_domains, one  # noqa: E402

RUN_DIR = ROOT / "output/eval/run-00-baseline-qwen25-vs-run04"
DEFAULT_BASE = RUN_DIR / "small-baseline-depth0.txt"
DEFAULT_RUN = RUN_DIR / "small-run04-depth2.txt"
DEFAULT_PUZZLES = ROOT / "kaggle/data/train-puzzles-small.jsonl"
DEFAULT_OUTPUT = RUN_DIR / "comparison.html"

TILES = ("XX", "NS", "EW", "NE", "NW", "SE", "SW")


def tile_svgs() -> dict[str, str]:
    """The reference game's track geometry, as data URIs (same as V2's assets)."""
    size = 45
    line = round(0.07 * size)
    inset = 0.25 * size
    sleeper_start = 0.15 * size
    sleeper_end = 0.85 * size
    olive = "#9e9e2b"
    black = "#000"

    straight = []
    for fraction in (0.125, 0.375, 0.625, 0.875):
        x = fraction * size
        straight.append(
            f'<line x1="{x}" y1="{sleeper_start}" x2="{x}" '
            f'y2="{sleeper_end}" stroke="{olive}" stroke-width="{line}"/>')
    for y in (size / 2 - inset, size / 2 + inset):
        straight.append(
            f'<line x1="0" y1="{y}" x2="{size}" y2="{y}" '
            f'stroke="{black}" stroke-width="{line}"/>')

    corner = []
    for index in range(4):
        angle = math.pi / 2 + math.pi / 16 + index * math.pi / 8
        x1 = size + sleeper_start * math.cos(angle)
        y1 = sleeper_start * math.sin(angle)
        x2 = size + sleeper_end * math.cos(angle)
        y2 = sleeper_end * math.sin(angle)
        corner.append(
            f'<line x1="{x1:.4f}" y1="{y1:.4f}" x2="{x2:.4f}" '
            f'y2="{y2:.4f}" stroke="{olive}" stroke-width="{line}"/>')
    for radius in (inset, size - inset):
        points = []
        for index in range(33):
            angle = math.pi / 2 + index * math.pi / 64
            points.append(f'{size + radius * math.cos(angle):.4f},'
                          f'{radius * math.sin(angle):.4f}')
        corner.append(f'<polyline points="{" ".join(points)}" fill="none" '
                      f'stroke="{black}" stroke-width="{line}"/>')

    parts = {
        "EW": "".join(straight),
        "NS": f'<g transform="rotate(90 {size / 2} {size / 2})">{"".join(straight)}</g>',
    }
    for name, angle in (("NE", 0), ("SE", 90), ("SW", 180), ("NW", 270)):
        parts[name] = (f'<g transform="rotate({angle} {size / 2} {size / 2})">'
                       f'{"".join(corner)}</g>')
    out = {}
    for name, body in parts.items():
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" '
               f'width="{size}" height="{size}">{body}</svg>')
        out[name] = "data:image/svg+xml;charset=utf-8," + quote(svg, safe="")
    return out


def load_records(path: Path) -> dict:
    records = {}
    for line in path.open(encoding="utf-8"):
        record = json.loads(line)
        records[record["puzzle_id"]] = record
    return records


def action_summary(action: dict) -> dict:
    """Keep only the fields the page renders."""
    kind = action["type"]
    out = {"type": kind}
    if kind in ("deduce", "branch"):
        out["cell"] = action["cell"]
    if kind == "deduce":
        out["candidates"] = action["candidates"]
        out["rule"] = action.get("rule")
    if kind == "branch":
        out["value"] = action["value"]
    if kind == "contradiction":
        out["rule"] = action.get("rule")
        out["anchor"] = action.get("anchor")
    if kind == "undo":
        out["to_step"] = action.get("to_step")
    if action.get("override"):
        out["override"] = True
        out["model_value"] = action.get("model_value")
        out["survivors"] = action.get("survivors")
    return out


def arm_events(puzzle: dict) -> tuple[list[dict], int]:
    """Valid actions plus a terminal ``invalid`` marker, and the invalid count."""
    events, invalid = [], 0
    for event in puzzle.get("history") or []:
        action = event.get("action")
        if event.get("valid") and action:
            entry = action_summary(action)
            search = event.get("branch_search") or {}
            if search.get("override"):
                entry["override"] = True
                entry["model_value"] = search.get("model_value")
                entry["survivors"] = search.get("survivors")
            events.append(entry)
        else:
            invalid += 1
            events.append({"type": "invalid", "raw": event.get("raw"),
                           "reason": event.get("reason")})
    return events, invalid


def replay(initial, events) -> list[list[int]]:
    board = [row[:] for row in initial]
    stack = []
    for action in events:
        kind = action["type"]
        if kind == "deduce":
            r, c = action["cell"]["row"] - 1, action["cell"]["column"] - 1
            board[r][c] = sum(BIT[name] for name in action["candidates"])
        elif kind == "branch":
            r, c = action["cell"]["row"] - 1, action["cell"]["column"] - 1
            stack.append({"board": [row[:] for row in board],
                          "cell": action["cell"], "value": action["value"]})
            board[r][c] = BIT[action["value"]]
        elif kind == "undo":
            checkpoint = stack.pop()
            board = [row[:] for row in checkpoint["board"]]
            r = checkpoint["cell"]["row"] - 1
            c = checkpoint["cell"]["column"] - 1
            board[r][c] &= ~BIT[checkpoint["value"]]
    return board


def build_arm(puzzle: dict, arm_name: str) -> dict:
    events, invalid = arm_events(puzzle)
    if len(events) != puzzle["actions"]:
        raise ValueError(f"{puzzle['puzzle_id']}/{arm_name}: "
                         f"{len(events)} events vs {puzzle['actions']} actions")
    return {
        "name": arm_name,
        "events": events,
        "invalid": invalid,
        "solved": puzzle["solved"],
        "cells": puzzle["correct_cells"],
        "total": puzzle["total_cells"],
        "actions": puzzle["actions"],
        "overrides": puzzle["search_overrides"],
        "failure": puzzle["failure"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--finetuned", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--puzzles", type=Path, default=DEFAULT_PUZZLES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    base = json.loads(args.baseline.read_text(encoding="utf-8"))
    run = json.loads(args.finetuned.read_text(encoding="utf-8"))
    records = load_records(args.puzzles)

    base_by_id = {p["puzzle_id"]: p for p in base["puzzles"]}
    run_by_id = {p["puzzle_id"]: p for p in run["puzzles"]}
    ids = sorted(base_by_id)

    puzzles = []
    checked = 0
    for pid in ids:
        record = records[pid]
        clue = clue_from_record(record)
        initial = initial_domains(clue)
        reference = [[tile or "XX" for tile in row] for row in record["solution"]]
        entry = {
            "id": pid,
            "rows": clue["rows"],
            "columns": clue["columns"],
            "givens": [{"row": r + 1, "column": c + 1, "value": shape}
                       for r, c, shape in clue["given_shapes"]],
            "initial": initial,
            "reference": reference,
            "arms": {"base": build_arm(base_by_id[pid], "base"),
                     "run": build_arm(run_by_id[pid], "run")},
        }
        # The page replays the same actions in JavaScript; verify that here.
        for key, puzzle in (("base", base_by_id[pid]), ("run", run_by_id[pid])):
            board = replay(initial, entry["arms"][key]["events"])
            prediction = puzzle["prediction"]
            for r, line in enumerate(board):
                for c, mask in enumerate(line):
                    want = prediction[r][c]
                    if want is None:
                        continue
                    if one(mask) != want:
                        raise ValueError(
                            f"{pid}/{key}: replay disagrees at r{r + 1}c{c + 1}: "
                            f"{one(mask)} vs {want}")
                    checked += 1
            if puzzle["solved"] and [[one(m) for m in row] for row in board] != reference:
                raise ValueError(f"{pid}/{key}: solved board does not equal the solution")
        puzzles.append(entry)

    puzzles.sort(key=lambda item: item["id"])
    # Open on the run-04-solved puzzle where the base model got furthest: it shows
    # the base policy solving for a while before it loses the output format.
    default_index = max(
        (index for index, item in enumerate(puzzles) if item["arms"]["run"]["solved"]),
        key=lambda index: puzzles[index]["arms"]["base"]["cells"])

    def median(values):
        values = sorted(values)
        return values[len(values) // 2]

    base_solved = sum(p["arms"]["base"]["solved"] for p in puzzles)
    run_solved = sum(p["arms"]["run"]["solved"] for p in puzzles)
    base_invalid = sum(p["arms"]["base"]["invalid"] for p in puzzles)
    run_invalid = sum(p["arms"]["run"]["invalid"] for p in puzzles)
    context = {
        "base_solved": base_solved,
        "run_solved": run_solved,
        "base_cells": sum(p["arms"]["base"]["cells"] for p in puzzles),
        "run_cells": sum(p["arms"]["run"]["cells"] for p in puzzles),
        "total_cells": sum(p["arms"]["base"]["total"] for p in puzzles),
        "base_invalid": base_invalid,
        "run_invalid": run_invalid,
        "base_median": median([p["arms"]["base"]["actions"] for p in puzzles]),
        "run_median": median([p["arms"]["run"]["actions"] for p in puzzles]),
        "run_overrides": sum(p["arms"]["run"]["overrides"] for p in puzzles),
        "count": len(puzzles),
        "base_model": base["run_id"],
        "run_model": run["run_id"],
        "default_index": default_index,
    }

    table_rows = []
    for index, item in enumerate(puzzles):
        b, r = item["arms"]["base"], item["arms"]["run"]
        def cell(arm):
            status = "solved" if arm["solved"] else (arm["failure"] or "failed")
            return (f'<td class="num">{arm["cells"]}/{arm["total"]}</td>'
                    f'<td class="num">{arm["actions"]}</td>'
                    f'<td class="num">{arm["invalid"]}</td>'
                    f'<td class="status {"yes" if arm["solved"] else "no"}">'
                    f'{html.escape(str(status))}</td>')
        table_rows.append(
            f'<tr data-index="{index}"><td><strong>{html.escape(item["id"].replace("-small", ""))}</strong></td>'
            f'{cell(b)}{cell(r)}</tr>')

    page = PAGE
    page = page.replace("__DATA__", json.dumps(
        {"puzzles": puzzles, "tiles": tile_svgs()}, separators=(",", ":")).replace("<", "\\u003c"))
    page = page.replace("__ROWS__", "\n".join(table_rows))
    for key, value in context.items():
        page = page.replace(f"__{key.upper()}__", html.escape(str(value)))
    if "__" in page.replace("__EASY", ""):
        leftover = [line for line in page.splitlines() if "__" in line and "data:image" not in line]
        raise SystemExit(f"unreplaced placeholder: {leftover[:3]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.2f} MB)")
    print(f"base {base_solved}/{len(puzzles)} ({context['base_cells']} cells, "
          f"{base_invalid} invalid) | run-04 {run_solved}/{len(puzzles)} "
          f"({context['run_cells']} cells, {run_invalid} invalid, "
          f"{context['run_overrides']} overrides)")
    print(f"replayed {checked:,} predicted cells and checked them against the reports")


PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Train Tracks: base model vs fine-tuned adapter</title>
<style>
:root {font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#192638;background:#f4f7fb}
* {box-sizing:border-box} body {margin:0} button,select,input {font:inherit}
header {background:#14263b;color:white;padding:28px max(18px,calc((100vw - 1320px)/2))}
h1 {font-size:clamp(24px,3vw,34px);margin:0 0 8px}
header p {margin:0;color:#d5e0ec;line-height:1.55;max-width:980px}
header p strong {color:white}
main {max-width:1360px;margin:auto;padding:20px 18px 48px}
.scoreboard {display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:18px}
.card {background:white;border:1px solid #dce5ee;border-radius:14px;padding:15px 18px}
.card h2 {margin:0 0 2px;font-size:13px;letter-spacing:.04em;text-transform:uppercase;color:#40536a}
.card .big {font-size:34px;font-weight:800;margin:2px 0 4px;font-variant-numeric:tabular-nums}
.card.base .big {color:#8b3030} .card.run .big {color:#17603d}
.card .sub {color:#53677c;font-size:13.5px;line-height:1.5}
.controls {display:flex;gap:10px;align-items:end;flex-wrap:wrap;margin-bottom:12px}
.controls label {display:grid;gap:4px;font-size:13px;font-weight:650;color:#40536a}
select {min-width:300px;max-width:100%;padding:9px 11px;border:1px solid #b7c7d7;border-radius:8px;background:white;color:#192638}
button {padding:9px 12px;border:1px solid #aabfd3;border-radius:8px;background:white;color:#1a3655}
button:hover {background:#edf3fa} button:disabled {opacity:.45}
.stepbar {display:flex;align-items:center;gap:12px;flex-wrap:wrap;background:white;border:1px solid #dce5ee;border-radius:12px;padding:12px 14px;margin-bottom:16px}
.stepbar input {flex:1;min-width:180px;accent-color:#265f9a}
#step-label {min-width:210px;font-variant-numeric:tabular-nums;font-weight:700;font-size:13.5px;color:#31485f}
.compare {display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start}
.panel {background:white;border:1px solid #dce5ee;border-radius:14px;padding:16px;min-width:0}
.panel.failed {border-color:#e9b6b2}
.panel.solved {border-color:#bfe3cd}
.panel h2 {font-size:17px;margin:0 0 6px}
.chips {display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 12px}
.chip {border-radius:99px;padding:4px 9px;font-size:12px;font-weight:750;background:#e6eef8;color:#20466f}
.chip.bad {background:#fbe2e0;color:#8b3030} .chip.good {background:#dcf4e3;color:#17603d}
.chip.mid {background:#fff0d2;color:#79510d}
.board {width:100%;overflow:auto} .board svg {display:block;max-width:100%;height:auto;margin:auto}
.detail {margin-top:10px;font-size:14.5px;line-height:1.55;color:#22344a;min-height:70px}
.detail .step {display:block;color:#53677c;font-size:12px;font-weight:750;letter-spacing:.03em;text-transform:uppercase;margin-bottom:3px}
.detail .muted {color:#6b7d92;font-size:13px}
.detail code, .callout code {background:#f0f4f9;border:1px solid #dde6ef;border-radius:6px;padding:1px 5px;font-size:13px}
.callout {margin-top:10px;border-radius:10px;padding:11px 13px;font-size:13.5px;line-height:1.55;background:#fbe2e0;border:1px solid #e9b6b2;color:#7c2a26}
.callout.ok {background:#eaf6ee;border-color:#bfe3cd;color:#1c5c39}
.callout code {background:rgba(255,255,255,.75);border-color:#e2b3ae}
.legend {display:flex;flex-wrap:wrap;gap:12px;margin:18px 0 0;color:#53677c;font-size:13px}
.legend span {display:inline-flex;align-items:center;gap:5px} .swatch {width:12px;height:12px;border-radius:3px;display:inline-block}
.given {background:#ccc;border:1px solid #a8b5c4}
.placed {background:linear-gradient(180deg,white 38%,#252b35 38%,#252b35 60%,white 60%);border:1px solid #a8b5c4}
.branch {background:#fff0cd;border:2px solid #b27600} .changed {background:#ddf3e5;border:2px solid #23834b}
.conflict {background:#fbe2e0;border:2px solid #c63530} .undo {background:#ece4fa;border:2px solid #7654b0}
.hint {color:#52677d;font-size:13px;margin:16px 0 0;line-height:1.55}
.evidence {margin-top:28px;background:white;border:1px solid #dce5ee;border-radius:14px;padding:16px}
.evidence summary {cursor:pointer;font-weight:750;font-size:16px}
table {width:100%;border-collapse:collapse;margin-top:12px;font-size:13.5px}
th,td {padding:7px 8px;border-bottom:1px solid #e6edf4;text-align:left}
th {color:#40536a;font-size:11.5px;letter-spacing:.04em;text-transform:uppercase;border-bottom:1px solid #d5e0ec}
td.num, th.num {text-align:right;font-variant-numeric:tabular-nums}
tbody tr {cursor:pointer} tbody tr:hover {background:#f7fafd}
.status.yes {color:#17603d;font-weight:750} .status.no {color:#8b3030;font-weight:750}
@media(max-width:980px){.compare,.scoreboard{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1>Same puzzle, two policies</h1>
  <p>Zero-shot <strong>Qwen2.5-1.5B</strong> on the left, the <strong>run-04</strong> LoRA adapter on the right. Same 32 small puzzles, same prompt and action menu, same 256-action budget; the fine-tuned arm also has the depth-2 branch filter. Step through both at once.</p>
  <p style="margin-top:8px">The base model puts <strong>312 of 1762 cells</strong> on the board but never finishes: one reply outside the action format ends the episode, and every board ends that way. The adapter finishes <strong>31 of 32</strong> with no invalid reply at all.</p>
</header>
<main>
  <section class="scoreboard">
    <article class="card base">
      <h2>Base Qwen2.5-1.5B</h2>
      <div class="big">__BASE_SOLVED__/__COUNT__</div>
      <div class="sub">__BASE_CELLS__/__TOTAL_CELLS__ cells · __BASE_INVALID__ replies outside the action format · median __BASE_MEDIAN__ actions</div>
    </article>
    <article class="card run">
      <h2>Fine-tuned · run-04 + filter</h2>
      <div class="big">__RUN_SOLVED__/__COUNT__</div>
      <div class="sub">__RUN_CELLS__/__TOTAL_CELLS__ cells · __RUN_INVALID__ invalid replies · median __RUN_MEDIAN__ actions · __RUN_OVERRIDES__ filter overrides</div>
    </article>
  </section>

  <div class="controls">
    <label for="puzzle">Puzzle<select id="puzzle"></select></label>
    <button id="previous" type="button">← Previous</button>
    <button id="next" type="button">Next →</button>
    <button id="jump-failure" type="button">Jump to the base failure</button>
  </div>
  <div class="stepbar">
    <input id="step" type="range" min="0" value="0" aria-label="Step both policies">
    <span id="step-label"></span>
  </div>

  <section class="compare">
    <article class="panel" id="panel-base">
      <h2>Base Qwen2.5-1.5B</h2>
      <div class="chips" id="chips-base"></div>
      <div class="board" id="board-base"></div>
      <div class="detail" id="detail-base"></div>
      <div id="callout-base"></div>
    </article>
    <article class="panel" id="panel-run">
      <h2>Fine-tuned · run-04</h2>
      <div class="chips" id="chips-run"></div>
      <div class="board" id="board-run"></div>
      <div class="detail" id="detail-run"></div>
      <div id="callout-run"></div>
    </article>
  </section>

  <div class="legend">
    <span><i class="swatch given"></i> Given track</span>
    <span><i class="swatch placed"></i> Fixed track</span>
    <span><i class="swatch changed"></i> Last deduction</span>
    <span><i class="swatch branch"></i> Trial branch</span>
    <span><i class="swatch conflict"></i> Contradiction</span>
    <span><i class="swatch undo"></i> Undo</span>
    <span>?N = N remaining choices</span>
  </div>
  <p class="hint">Green clue numbers are satisfied. The board shows the state after the selected step; each policy stops on its own last step, so once the left panel is done the right one keeps going. Click a row in the table to jump to that puzzle.</p>

  <details class="evidence">
    <summary>All 32 puzzles, both policies</summary>
    <table>
      <thead>
        <tr>
          <th rowspan="2">Puzzle</th>
          <th class="num" colspan="4">Base Qwen2.5-1.5B</th>
          <th class="num" colspan="4">Fine-tuned · run-04</th>
        </tr>
        <tr>
          <th class="num">cells</th><th class="num">actions</th><th class="num">invalid</th><th>outcome</th>
          <th class="num">cells</th><th class="num">actions</th><th class="num">invalid</th><th>outcome</th>
        </tr>
      </thead>
      <tbody>
__ROWS__
      </tbody>
    </table>
  </details>
</main>
<script>
const DATA=__DATA__;
const PUZZLES=DATA.puzzles;
const TILE_IMAGES=DATA.tiles;
const TILES=['XX','NS','EW','NE','NW','SE','SW'];
const BITS=Object.fromEntries(TILES.map((name,i)=>[name,1<<i]));
const DIR={N:[0,-1],E:[1,0],S:[0,1],W:[-1,0]};
const els=Object.fromEntries(['puzzle','step','step-label','previous','next','jump-failure',
  'board-base','board-run','chips-base','chips-run','detail-base','detail-run',
  'callout-base','callout-run','panel-base','panel-run'].map(id=>[id,document.getElementById(id)]));
let selected=0, position=0, frames={base:[],run:[]};
function copy(board){return board.map(line=>line.slice())}
function names(mask){return TILES.filter(tile=>mask&BITS[tile])}
function escapeText(value){return String(value).replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]))}
function buildFrames(arm){
  const puzzle=PUZZLES[selected];
  let board=copy(puzzle.initial),stack=[];
  const frames=[{board:copy(board),stack:[],action:null}];
  for(const action of arm.events){
    if(action.type==='deduce'){
      const r=action.cell.row-1,c=action.cell.column-1;
      board[r][c]=action.candidates.reduce((mask,name)=>mask|BITS[name],0);
    }else if(action.type==='branch'){
      const r=action.cell.row-1,c=action.cell.column-1;
      stack.push({board:copy(board),cell:action.cell,value:action.value});
      board[r][c]=BITS[action.value];
    }else if(action.type==='undo'){
      const checkpoint=stack.pop();
      if(checkpoint){
        board=copy(checkpoint.board);
        const r=checkpoint.cell.row-1,c=checkpoint.cell.column-1;
        board[r][c]&=~BITS[checkpoint.value];
      }
    }
    frames.push({board:copy(board),stack:stack.map(x=>({cell:x.cell,value:x.value})),action});
  }
  return frames;
}
function highlight(action){
  if(!action)return null;
  if(action.type==='undo')return null;
  if(action.cell)return {row:action.cell.row,column:action.cell.column,type:action.type};
  const a=action.anchor;
  if(a&&(a.row!==undefined||a.column!==undefined))return {anchor:a,type:action.type};
  return null;
}
function anchorText(action){
  const a=action.anchor; if(!a)return 'the board';
  if(a.row!==undefined&&a.column!==undefined)return `r${a.row}c${a.column}`;
  if(a.row!==undefined)return `row ${a.row} (clue ${a.clue})`;
  if(a.column!==undefined)return `column ${a.column} (clue ${a.clue})`;
  return 'the board';
}
function svgBoard(puzzle,frame,action){
  const size=45,left=22,top=43,right=48,bottom=23;
  const board=frame.board,h=board.length,w=board[0].length;
  const W=left+w*size+right,H=top+h*size+bottom;
  const given=new Set(puzzle.givens.map(g=>`${g.row},${g.column}`));
  const active=highlight(action);
  let parts=[`<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Puzzle board after the selected step" xmlns="http://www.w3.org/2000/svg">`,
    `<rect width="${W}" height="${H}" fill="white"/>`];
  puzzle.columns.forEach((clue,c)=>{
    const confirmed=board.reduce((count,line)=>count+(!(line[c]&BITS.XX)?1:0),0);
    parts.push(`<text x="${left+(c+.5)*size}" y="29" text-anchor="middle" font-size="19" font-weight="700" fill="${confirmed===clue?'#008800':'#243242'}">${clue}</text>`);
  });
  puzzle.rows.forEach((clue,r)=>{
    const confirmed=board[r].filter(mask=>!(mask&BITS.XX)).length;
    parts.push(`<text x="${left+w*size+24}" y="${top+(r+.5)*size+6}" text-anchor="middle" font-size="19" font-weight="700" fill="${confirmed===clue?'#008800':'#243242'}">${clue}</text>`);
  });
  for(let r=0;r<h;r++)for(let c=0;c<w;c++){
    const x=left+c*size,y=top+r*size,cx=x+size/2,cy=y+size/2;
    const values=names(board[r][c]),fixed=values.length===1,cellKey=`${r+1},${c+1}`;
    let fill=given.has(cellKey)?'#ccc':'white';
    if(active){
      if(active.row===r+1&&active.column===c+1){
        fill=active.type==='branch'?'#fff0cd':active.type==='contradiction'?'#fbe2e0':active.type==='undo'?'#ece4fa':'#ddf3e5';
      }else if(active.anchor){
        const a=active.anchor;
        if((a.row===r+1&&a.column===undefined)||(a.column===c+1&&a.row===undefined))fill='#fbe2e0';
      }
    }
    parts.push(`<g><title>row ${r+1}, column ${c+1}: ${escapeText(values.join(', '))}</title><rect x="${x}" y="${y}" width="${size}" height="${size}" fill="${fill}" stroke="#a8b5c4" stroke-width="1"/>`);
    if(fixed&&values[0]==='XX'){
      parts.push(`<path d="M${cx-7} ${cy-7}L${cx+7} ${cy+7}M${cx-7} ${cy+7}L${cx+7} ${cy-7}" fill="none" stroke="#8b929c" stroke-width="2" stroke-linecap="round"/>`);
    }else if(fixed){
      parts.push(`<image href="${TILE_IMAGES[values[0]]}" x="${x}" y="${y}" width="${size}" height="${size}" preserveAspectRatio="xMidYMid meet"/>`);
      if(active&&active.row===r+1&&active.column===c+1&&fill!=='white'){
        const border=active.type==='branch'?'#b27600':active.type==='contradiction'?'#c63530':'#23834b';
        parts.push(`<rect x="${x+2}" y="${y+2}" width="${size-4}" height="${size-4}" fill="none" stroke="${border}" stroke-width="3"/>`);
      }
    }else{
      parts.push(`<text x="${cx}" y="${cy+5}" text-anchor="middle" font-size="16" font-weight="650" fill="#956819">?${values.length}</text>`);
    }
    parts.push('</g>');
  }
  parts.push('</svg>');
  return parts.join('');
}
function describe(puzzle,key,arm,frameIndex){
  const armFrames=frames[key],frame=armFrames[frameIndex];
  const action=frame.action;
  const html=[];
  if(!action){
    return {label:'Start',text:'Initial clues and given tracks. No action taken yet.'};
  }
  const at=action.cell?`r${action.cell.row}c${action.cell.column}`:'';
  if(action.type==='deduce'){
    const before=armFrames[frameIndex-1].board[action.cell.row-1][action.cell.column-1];
    const after=frame.board[action.cell.row-1][action.cell.column-1];
    const removed=names(before).filter(name=>!(after&BITS[name]));
    return {label:'Deduction',
      text:`Remove ${escapeText(removed.join(', '))} at <strong>${at}</strong>; remaining ${escapeText(names(after).join(', '))}.${action.rule?` <span class="muted">Rule: ${escapeText(action.rule.replaceAll('_',' '))}</span>`:''}`};
  }
  if(action.type==='branch'){
    let text=`Trial branch: <strong>${action.value}</strong> at <strong>${at}</strong>.`;
    if(action.override)text+=` The filter rejected the model's <strong>${escapeText(action.model_value)}</strong> because propagation refuted it.`;
    return {label:action.override?'Branch · filter override':'Trial branch',text};
  }
  if(action.type==='contradiction'){
    return {label:'Contradiction',
      text:`The trial cannot satisfy <strong>${escapeText((action.rule||'a rule').replaceAll('_',' '))}</strong> at ${escapeText(anchorText(action))}. The board is refuted and will be rolled back.`};
  }
  if(action.type==='undo'){
    const branch=arm.events[action.to_step];
    const where=branch&&branch.cell?`r${branch.cell.row}c${branch.cell.column}`:'the trial';
    const value=branch?branch.value:'';
    return {label:'Undo',
      text:`Restore the board from before the <strong>${escapeText(value)}</strong> trial at <strong>${where}</strong>, and rule that value out.`};
  }
  if(action.type==='finish')return {label:'Finish',text:'The board satisfies every clue and forms one connected track.'};
  if(action.type==='invalid'){
    return {label:'Out of format',
      text:`The model replied <code>${escapeText(action.raw)}</code>. The harness expects <code>{"choice":"aNNN"}</code>, so no action is applied and the episode ends.`};
  }
  return {label:action.type,text:''};
}
function chips(key,arm,frame){
  const fixed=frame.board.flat().filter(mask=>(mask&(mask-1))===0).length;
  const pieces=[];
  const status=arm.solved?'solved':(arm.failure||'failed');
  pieces.push(`<span class="chip ${arm.solved?'good':(arm.invalid?'bad':'mid')}">final: ${escapeText(status)}</span>`);
  pieces.push(`<span class="chip">${arm.cells}/${arm.total} cells</span>`);
  pieces.push(`<span class="chip">${fixed}/${arm.total} fixed now</span>`);
  pieces.push(`<span class="chip">${arm.actions} actions</span>`);
  if(arm.invalid)pieces.push(`<span class="chip bad">${arm.invalid} invalid</span>`);
  if(arm.overrides)pieces.push(`<span class="chip mid">${arm.overrides} overrides</span>`);
  return pieces.join('');
}
function renderPanel(key){
  const puzzle=PUZZLES[selected],arm=puzzle.arms[key],armFrames=frames[key];
  const index=Math.min(position,armFrames.length-1);
  const frame=armFrames[index];
  els[`board-${key}`].innerHTML=svgBoard(puzzle,frame,frame.action);
  els[`chips-${key}`].innerHTML=chips(key,arm,frame);
  const described=describe(puzzle,key,arm,index);
  els[`detail-${key}`].innerHTML=`<span class="step">Step ${index} of ${arm.events.length} · ${escapeText(described.label)}</span>${described.text}`;
  els[`panel-${key}`].className='panel '+(arm.solved?'solved':(index>=arm.events.length?'failed':''));
  const callout=els[`callout-${key}`];
  const failure=arm.events.findIndex(event=>event.type==='invalid');
  if(failure>=0 && index>=failure+1){
    callout.innerHTML=`<div class="callout"><b>The episode stops here.</b> One reply outside the action format is enough: <code>${escapeText(arm.events[failure].raw)}</code> instead of <code>{"choice":"aNNN"}</code>. This is a format failure, not a puzzle failure — the board is at ${arm.cells}/${arm.total} cells.</div>`;
  }else if(!arm.solved && index>=arm.events.length){
    callout.innerHTML=`<div class="callout"><b>${escapeText(arm.failure||'failed')}</b> after ${arm.actions} actions, ${arm.cells}/${arm.total} cells correct.</div>`;
  }else if(arm.solved && index>=arm.events.length){
    callout.innerHTML=`<div class="callout ok"><b>Solved</b> in ${arm.actions} actions, ${arm.cells}/${arm.total} cells on the audited solution.</div>`;
  }else{
    callout.innerHTML='';
  }
}
function render(){
  const puzzle=PUZZLES[selected];
  const baseLen=puzzle.arms.base.events.length,runLen=puzzle.arms.run.events.length;
  const maxLen=Math.max(baseLen,runLen);
  position=Math.min(position,maxLen);
  els.step.max=maxLen;
  els.step.value=position;
  const failureIndex=puzzle.arms.base.events.findIndex(event=>event.type==='invalid');
  const where=failureIndex>=0?failureIndex+1:baseLen;
  els['step-label'].textContent=`Step ${position} of ${maxLen} · base ${Math.min(position,baseLen)}/${baseLen} · fine-tuned ${Math.min(position,runLen)}/${runLen}`;
  els.previous.disabled=position===0;
  els.next.disabled=position===maxLen;
  els['jump-failure'].textContent=`Jump to the base failure (step ${where})`;
  renderPanel('base');
  renderPanel('run');
}
function choosePuzzle(index){
  selected=index;position=0;
  const puzzle=PUZZLES[selected];
  frames={base:buildFrames(puzzle.arms.base),run:buildFrames(puzzle.arms.run)};
  els.puzzle.value=String(index);
  render();
}
function fillPuzzles(){
  els.puzzle.replaceChildren();
  PUZZLES.forEach((puzzle,index)=>{
    const option=document.createElement('option');
    option.value=index;
    const b=puzzle.arms.base,r=puzzle.arms.run;
    option.textContent=`${puzzle.id.replace('-small','')} · base ${b.cells}/${b.total}${b.solved?' ✓':' ✗'} · fine-tuned ${r.solved?'solved':'failed'} (${r.actions} actions)`;
    els.puzzle.append(option);
  });
}
els.puzzle.addEventListener('change',()=>choosePuzzle(Number(els.puzzle.value)));
els.step.addEventListener('input',()=>{position=Number(els.step.value);render()});
els.previous.addEventListener('click',()=>{if(position>0){position--;render()}});
els.next.addEventListener('click',()=>{if(position<Number(els.step.max)){position++;render()}});
els['jump-failure'].addEventListener('click',()=>{
  const baseLen=PUZZLES[selected].arms.base.events.length;
  const failure=PUZZLES[selected].arms.base.events.findIndex(event=>event.type==='invalid');
  position=failure>=0?failure+1:baseLen;render();
});
document.addEventListener('keydown',event=>{
  if(['SELECT','INPUT','BUTTON','TEXTAREA'].includes(document.activeElement.tagName))return;
  if(event.key==='ArrowRight'&&position<Number(els.step.max)){position++;render()}
  if(event.key==='ArrowLeft'&&position>0){position--;render()}
});
document.querySelectorAll('tbody tr').forEach(row=>{
  row.addEventListener('click',()=>{
    choosePuzzle(Number(row.dataset.index));
    document.querySelector('.controls').scrollIntoView({behavior:'smooth',block:'start'});
  });
});
fillPuzzles();
choosePuzzle(__DEFAULT_INDEX__);
</script>
</body>
</html>
'''


if __name__ == "__main__":
    main()
