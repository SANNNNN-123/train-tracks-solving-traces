"""Build a standalone predicted-vs-reference board viewer from a Modal rollout.

Run: python scripts/render_eval_comparison.py output/eval/RUN/rollout.json
"""

import argparse
import json
from pathlib import Path


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Train Tracks evaluation: predicted vs actual</title>
<style>
:root{font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#1c2b3d;background:#f4f7fb}
*{box-sizing:border-box}body{margin:0}header{background:#14263b;color:white;padding:24px max(18px,calc((100vw - 1200px)/2))}
h1{margin:0 0 8px;font-size:clamp(24px,3vw,34px)}header p{margin:0;color:#d6e1ed}
main{max-width:1200px;margin:auto;padding:20px 18px 42px}.controls{display:flex;gap:10px;align-items:end;flex-wrap:wrap;margin-bottom:16px}
label{display:grid;gap:4px;font-size:13px;font-weight:650;color:#496078}select,button{font:inherit;background:white;color:#1c2b3d;border:1px solid #b7c7d7;border-radius:8px;padding:9px 12px}
select{min-width:280px;max-width:100%}button{cursor:pointer}button:disabled{opacity:.45;cursor:default}
.status{background:white;border:1px solid #dce5ee;border-radius:12px;padding:13px 16px;margin-bottom:16px;line-height:1.6}
.boards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.panel{background:white;border:1px solid #dce5ee;border-radius:14px;padding:14px;min-width:0;overflow:auto}
.panel h2{font-size:18px;margin:0 0 10px}.board{border-collapse:collapse;margin:auto}.board th{font-size:13px;color:#52677d;min-width:25px;text-align:center}
.board td{width:49px;height:49px;min-width:49px;border:1px solid #cad5df;text-align:center;position:relative;background:#fff;color:#273b50}
.board td.bad{background:#ffe5e0;color:#a52b25}.board svg{width:33px;height:33px;display:block;margin:auto;stroke:currentColor;stroke-width:11;stroke-linecap:round;stroke-linejoin:round;fill:none}
.board .unknown{font-size:24px;font-weight:700}.board .tile-name{position:absolute;bottom:1px;right:2px;font-size:9px;font-weight:700;color:#5c6d7b}
.board td.bad .tile-name{color:#9a3b35}.note{color:#52677d;font-size:13px;margin:12px 0 0}
details{background:white;border:1px solid #dce5ee;border-radius:12px;padding:11px 16px;margin:-4px 0 16px}summary{cursor:pointer;font-weight:650}pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:10px 0 0;font-size:12px}
@media(max-width:850px){.boards{grid-template-columns:1fr}}
</style></head><body>
<header><h1>Predicted vs actual boards</h1><p id="overview"></p></header>
<main><div class="controls"><label>Puzzle<select id="puzzle"></select></label><button id="previous">Previous</button><button id="next">Next</button></div>
<div class="status" id="status"></div><details><summary>Rejected model action</summary><pre id="rejected-action"></pre></details>
<div class="boards"><section class="panel"><h2>Model board when stopped</h2><div id="predicted"></div></section>
<section class="panel"><h2>Reference solution</h2><div id="reference"></div></section></div>
<p class="note">Red cells differ from the reference. A ? means the model stopped before choosing a single tile for that cell. Row and column numbers are puzzle clues. Correct cells can include given tiles.</p></main>
<script>
const DATA=__DATA__;
const PATHS={NS:'M50 0 V100',EW:'M0 50 H100',NE:'M50 0 V50 H100',NW:'M50 0 V50 H0',SE:'M100 50 H50 V100',SW:'M0 50 H50 V100'};
const escapeText=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const select=document.getElementById('puzzle');let selected=0;
DATA.puzzles.forEach((p,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${i+1}. ${p.id}`;select.appendChild(o)});
document.getElementById('overview').textContent=`${DATA.attempted} puzzles · ${DATA.solved} solved · ${DATA.correct_cells}/${DATA.total_cells} cells resolved to the reference`;
function tile(value){if(!value)return '<span class="unknown">?</span>';const path=PATHS[value];return path?`<svg viewBox="0 0 100 100" aria-hidden="true"><path d="${path}"/></svg><span class="tile-name">${value}</span>`:`<span class="tile-name">${escapeText(value)}</span>`}
function board(p,actual){let out='<table class="board"><thead><tr><th></th>';for(const clue of p.columns)out+=`<th>${clue}</th>`;out+='</tr></thead><tbody>';
for(let r=0;r<p.reference.length;r++){out+=`<tr><th>${p.rows[r]??''}</th>`;for(let c=0;c<p.reference[r].length;c++){const predicted=p.prediction[r][c],reference=p.reference[r][c];const value=actual?reference:predicted;const bad=!actual&&predicted!==reference;out+=`<td class="${bad?'bad':''}" title="row ${r+1}, column ${c+1}: predicted ${predicted??'?'}; actual ${reference}">${tile(value)}</td>`}out+='</tr>'}return out+'</tbody></table>'}
function render(){const p=DATA.puzzles[selected];select.value=selected;document.getElementById('status').innerHTML=`<strong>${escapeText(p.id)}</strong> · ${p.correct_cells}/${p.total_cells} cells resolved to the reference · ${p.valid_actions} accepted of ${p.actions} proposed actions · ${p.solved?'Solved':'Stopped: '+escapeText(p.failure||'unknown reason')}`;
document.getElementById('rejected-action').textContent=p.rejected_action?JSON.stringify(p.rejected_action,null,2):'No rejected action recorded';
document.getElementById('predicted').innerHTML=board(p,false);document.getElementById('reference').innerHTML=board(p,true);
document.getElementById('previous').disabled=selected===0;document.getElementById('next').disabled=selected===DATA.puzzles.length-1}
select.addEventListener('change',()=>{selected=Number(select.value);render()});document.getElementById('previous').onclick=()=>{selected--;render()};document.getElementById('next').onclick=()=>{selected++;render()};render();
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout", type=Path)
    args = parser.parse_args()
    report = json.loads(args.rollout.read_text(encoding="utf-8"))
    puzzles = []
    for item in report["puzzles"]:
        history = item.get("history") or []
        state = history[0]["state"] if history else {}
        rejected = next((event.get("action") for event in history if not event.get("valid")), None)
        puzzles.append({
            "id": item["puzzle_id"], "prediction": item["prediction"],
            "reference": item["reference"], "rows": state.get("row_clues", []),
            "columns": state.get("column_clues", []), "correct_cells": item["correct_cells"],
            "total_cells": item["total_cells"], "actions": item["actions"],
            "valid_actions": sum(bool(event.get("valid")) for event in history),
            "rejected_action": rejected,
            "failure": item.get("failure"), "solved": item["solved"],
        })
    data = {"attempted": report["attempted"], "solved": report["solved"],
            "correct_cells": report["correct_cells"], "total_cells": report["total_cells"],
            "puzzles": puzzles}
    embedded = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("<", "\\u003c")
    page = PAGE.replace("__DATA__", embedded)
    output = args.rollout.with_name("predicted-vs-actual.html")
    output.write_text(page, encoding="utf-8")
    print(f"Wrote {output} ({len(puzzles)} puzzles)")


if __name__ == "__main__":
    main()
