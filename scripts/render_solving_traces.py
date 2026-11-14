"""Build a standalone HTML step-through viewer for offline solving traces.

Run from the repository root:
    python modal/tools/render_solving_traces.py
"""

from __future__ import annotations

import argparse
import html
import json
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/train.jsonl"
DEFAULT_OUTPUT = ROOT / "solving-traces.html"


def compact_traces(path: Path) -> list[dict]:
    puzzles = OrderedDict()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            puzzle_id = row["puzzle_id"]
            if puzzle_id not in puzzles:
                state = row["state"]
                puzzles[puzzle_id] = {
                    "id": puzzle_id,
                    "rows": state["row_clues"],
                    "columns": state["column_clues"],
                    "givens": state["givens"],
                    "initial": state["domains"],
                    "events": [],
                }
            puzzle = puzzles[puzzle_id]
            if row["step_index"] != len(puzzle["events"]):
                raise ValueError(f"Nonsequential trace: {puzzle_id}")
            action = dict(row["action"])
            if action["type"] == "branch":
                action["expert"] = row["expert_policy_label"]
            puzzle["events"].append(action)
    if not puzzles:
        raise ValueError("Trace file is empty")
    for puzzle in puzzles.values():
        if puzzle["events"][-1]["type"] != "finish":
            raise ValueError(f"Trace does not finish: {puzzle['id']}")
    return list(puzzles.values())


PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Train Tracks solving traces</title>
<style>
:root {font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#192638;background:#f4f7fb}
* {box-sizing:border-box} body {margin:0} button,select,input {font:inherit}
header {background:#14263b;color:white;padding:24px max(18px,calc((100vw - 1180px)/2))}
h1 {font-size:clamp(24px,3vw,34px);margin:0 0 6px} header p {margin:0;color:#d5e0ec;line-height:1.45}
main {max-width:1220px;margin:auto;padding:20px 18px 42px}
.controls {display:flex;gap:10px;align-items:end;flex-wrap:wrap;margin-bottom:16px}
.controls label {display:grid;gap:4px;font-size:13px;font-weight:650;color:#40536a}
select {min-width:220px;max-width:100%;padding:9px 11px;border:1px solid #b7c7d7;border-radius:8px;background:white;color:#192638}
button {padding:9px 12px;border:1px solid #aabfd3;border-radius:8px;background:white;color:#1a3655}
button:hover {background:#edf3fa} button:disabled {opacity:.45}
.stepbar {display:flex;align-items:center;gap:12px;flex-wrap:wrap;background:white;border:1px solid #dce5ee;border-radius:12px;padding:12px 14px;margin-bottom:16px}
.stepbar input {flex:1;min-width:180px;accent-color:#265f9a} #step-label {min-width:100px;font-variant-numeric:tabular-nums;font-weight:700}
.layout {display:grid;grid-template-columns:minmax(0,1fr) minmax(260px,340px);gap:18px;align-items:start}
.board-panel,.detail {background:white;border:1px solid #dce5ee;border-radius:14px;padding:16px}
.board-panel {min-width:0} #board {width:100%;overflow:auto} #board svg {display:block;width:100%;max-width:700px;margin:auto}
.legend {display:flex;flex-wrap:wrap;gap:12px;margin:12px 0 0;color:#53677c;font-size:13px}
.legend span {display:inline-flex;align-items:center;gap:5px} .swatch {width:12px;height:12px;border-radius:3px;display:inline-block}
.given {background:#245baa}.placed {background:#252b35}.branch {background:#edab34}.changed {background:#8bcfa8}.conflict {background:#e57b78}
.detail h2 {font-size:18px;margin:0 0 12px} .badge {display:inline-block;border-radius:99px;padding:4px 9px;font-size:12px;font-weight:750;background:#e6eef8;color:#20466f}
.badge.wrong {background:#fff0d2;color:#79510d}.badge.conflict {background:#fbe2e0;color:#8b3030}.badge.undo {background:#ece4fa;color:#5b3c89}.badge.done {background:#dcf4e3;color:#17603d}
#action {font-size:17px;line-height:1.48;margin:13px 0 10px;overflow-wrap:anywhere}
#meta {color:#53677c;line-height:1.55;font-size:14px} #candidates {margin-top:13px;padding-top:12px;border-top:1px solid #e3eaf1;color:#31485f;font-size:14px;line-height:1.5}
.hint {color:#52677d;font-size:13px;margin:14px 0 0}
@media(max-width:850px) {.layout {grid-template-columns:1fr}.detail {order:-1}}
@media(max-width:520px) {.controls label {width:100%}select {width:100%}.stepbar {gap:8px}.stepbar input {order:2;flex-basis:100%}}
</style>
</head>
<body>
<header><h1>Train Tracks solving traces</h1><p>Step through the offline teacher's deductions, trial branches, contradictions, and undos across __PUZZLE_COUNT__ training puzzles.</p></header>
<main>
<div class="controls">
  <label for="puzzle">Puzzle<select id="puzzle"></select></label>
  <button id="next-branch" type="button">Next branch</button>
  <button id="next-conflict" type="button">Next contradiction</button>
</div>
<div class="stepbar">
  <button id="previous" type="button" aria-label="Previous action">← Previous</button>
  <button id="next" type="button" aria-label="Next action">Next →</button>
  <span id="step-label"></span>
  <input id="step" type="range" min="0" value="0" aria-label="Trace step">
</div>
<div class="layout">
  <section class="board-panel" aria-label="Puzzle board">
    <div id="board"></div>
    <div class="legend">
      <span><i class="swatch given"></i> Given rail</span>
      <span><i class="swatch placed"></i> Fixed rail</span>
      <span><i class="swatch changed"></i> Deduction</span>
      <span><i class="swatch branch"></i> Trial branch</span>
      <span><i class="swatch conflict"></i> Contradiction</span>
      <span>× Empty · ?N N candidates</span>
    </div>
  </section>
  <aside class="detail" aria-live="polite">
    <h2 id="puzzle-title"></h2>
    <span class="badge" id="kind">Start</span>
    <p id="action"></p>
    <div id="meta"></div>
    <div id="candidates"></div>
  </aside>
</div>
<p class="hint">The board shows the state after the selected action. Use the slider or the left and right arrow keys. Hover over an unknown cell to see its remaining tiles.</p>
</main>
<script>
const PUZZLES=__DATA__;
const TILES=['XX','NS','EW','NE','NW','SE','SW'];
const BITS=Object.fromEntries(TILES.map((name,i)=>[name,1<<i]));
const DIR={N:[0,-1],E:[1,0],S:[0,1],W:[-1,0]};
const els=Object.fromEntries(['puzzle','step','step-label','previous','next','next-branch','next-conflict','board','puzzle-title','kind','action','meta','candidates'].map(id=>[id,document.getElementById(id)]));
let selected=0,position=0,frames=[];
function copy(board){return board.map(line=>line.slice())}
function names(mask){return TILES.filter(tile=>mask&BITS[tile])}
function escapeText(value){return String(value).replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]))}
function buildFrames(puzzle){
  let board=copy(puzzle.initial),stack=[];
  const result=[{board:copy(board),stack:[]}];
  puzzle.events.forEach((action,index)=>{
    if(action.type==='deduce'){
      const r=action.cell.row-1,c=action.cell.column-1;
      board[r][c]=action.candidates.reduce((mask,name)=>mask|BITS[name],0);
    }else if(action.type==='branch'){
      stack.push({step:index,board:copy(board),cell:action.cell,value:action.value});
      board[action.cell.row-1][action.cell.column-1]=BITS[action.value];
    }else if(action.type==='undo'){
      const checkpoint=stack.pop();
      if(!checkpoint||checkpoint.step!==action.to_step)throw Error('Invalid undo in trace');
      board=copy(checkpoint.board);
    }
    result.push({board:copy(board),stack:stack.map(x=>({step:x.step,cell:x.cell,value:x.value}))});
  });
  return result;
}
function actionAnchor(action){
  if(!action.anchor)return '';
  const a=action.anchor;
  if(a.row!==undefined&&a.column!==undefined)return `cell r${a.row} c${a.column}`;
  if(a.row!==undefined)return `row ${a.row} (clue ${a.clue})`;
  if(a.column!==undefined)return `column ${a.column} (clue ${a.clue})`;
  return 'completed board';
}
function currentHighlight(action,puzzle){
  if(!action)return null;
  if(action.type==='undo'){
    const branch=puzzle.events[action.to_step];
    return branch&&branch.cell?{row:branch.cell.row,column:branch.cell.column,type:'undo'}:null;
  }
  if(action.cell)return {row:action.cell.row,column:action.cell.column,type:action.type};
  return {anchor:action.anchor,type:action.type};
}
function svgBoard(puzzle,board,action){
  const size=48,left=22,top=43,right=48,bottom=23;
  const h=board.length,w=board[0].length,W=left+w*size+right,H=top+h*size+bottom;
  const given=new Set(puzzle.givens.map(g=>`${g.row},${g.column}`));
  const active=currentHighlight(action,puzzle);
  let parts=[`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Puzzle board after selected action" xmlns="http://www.w3.org/2000/svg">`,
    `<rect width="${W}" height="${H}" fill="white"/>`];
  puzzle.columns.forEach((clue,c)=>parts.push(`<text x="${left+(c+.5)*size}" y="29" text-anchor="middle" font-size="19" font-weight="700" fill="#243242">${clue}</text>`));
  puzzle.rows.forEach((clue,r)=>parts.push(`<text x="${left+w*size+24}" y="${top+(r+.5)*size+6}" text-anchor="middle" font-size="19" font-weight="700" fill="#243242">${clue}</text>`));
  for(let r=0;r<h;r++)for(let c=0;c<w;c++){
    const x=left+c*size,y=top+r*size,cx=x+size/2,cy=y+size/2;
    const values=names(board[r][c]),fixed=values.length===1,cellKey=`${r+1},${c+1}`;
    let fill='white';
    if(active){
      const a=active.anchor;
      if(active.row===r+1&&active.column===c+1)fill=active.type==='branch'?'#fff0cd':active.type==='contradiction'?'#fbe2e0':active.type==='undo'?'#ece4fa':'#ddf3e5';
      else if(a&&((a.row===r+1&&a.column===undefined)||(a.column===c+1&&a.row===undefined)))fill='#fbe2e0';
    }
    parts.push(`<g><title>row ${r+1}, column ${c+1}: ${escapeText(values.join(', '))}</title><rect x="${x}" y="${y}" width="${size}" height="${size}" fill="${fill}" stroke="#a8b5c4" stroke-width="1"/>`);
    if(fixed&&values[0]==='XX'){
      parts.push(`<path d="M${cx-7} ${cy-7}L${cx+7} ${cy+7}M${cx-7} ${cy+7}L${cx+7} ${cy-7}" fill="none" stroke="#8b929c" stroke-width="2" stroke-linecap="round"/>`);
    }else if(fixed){
      const color=given.has(cellKey)?'#245baa':'#252b35';
      for(const d of values[0]){
        const [dx,dy]=DIR[d];
        parts.push(`<line x1="${cx}" y1="${cy}" x2="${cx+dx*size/2}" y2="${cy+dy*size/2}" stroke="${color}" stroke-width="4" stroke-linecap="round"/>`);
      }
      if(given.has(cellKey))parts.push(`<circle cx="${cx}" cy="${cy}" r="3" fill="#245baa"/>`);
    }else{
      parts.push(`<text x="${cx}" y="${cy+5}" text-anchor="middle" font-size="16" font-weight="650" fill="#956819">?${values.length}</text>`);
    }
    parts.push('</g>');
  }
  parts.push('</svg>');
  return parts.join('');
}
function describe(puzzle,frame,action,step){
  els.kind.className='badge';
  if(!action){els.kind.textContent='Start';els.action.textContent='Initial clues and given rails. No actions have been taken.'}
  else if(action.type==='deduce'){
    els.kind.textContent='Deduction';
    const before=frames[step-1].board[action.cell.row-1][action.cell.column-1];
    const after=frame.board[action.cell.row-1][action.cell.column-1];
    const removed=names(before).filter(name=>!(after&BITS[name]));
    els.action.textContent=`At row ${action.cell.row}, column ${action.cell.column}, remove ${removed.join(', ')}. Remaining: ${names(after).join(', ')}.`;
  }else if(action.type==='branch'){
    els.kind.className='badge'+(action.expert?'':' wrong');
    els.kind.textContent=action.expert?'Branch':'Wrong trial branch';
    els.action.textContent=`Try ${action.value} at row ${action.cell.row}, column ${action.cell.column}.`;
  }else if(action.type==='contradiction'){
    els.kind.className='badge conflict';els.kind.textContent='Contradiction';
    els.action.textContent=`The branch cannot satisfy ${action.rule.replaceAll('_',' ')} at ${actionAnchor(action)}.`;
  }else if(action.type==='undo'){
    els.kind.className='badge undo';els.kind.textContent='Undo';
    const branch=puzzle.events[action.to_step];
    els.action.textContent=`Restore the board from before the ${branch.value} trial at row ${branch.cell.row}, column ${branch.cell.column}.`;
  }else{els.kind.className='badge done';els.kind.textContent='Solved';els.action.textContent='The complete board satisfies all clues and forms one connected track.'}
  const h=frame.board.length,w=frame.board[0].length;
  const fixed=frame.board.flat().filter(mask=>(mask&(mask-1))===0).length;
  els.meta.textContent=`${fixed}/${h*w} cells fixed · ${frame.stack.length} active branches${action&&action.rule?' · Rule: '+action.rule.replaceAll('_',' '):''}`;
  if(action&&action.cell){
    const mask=frame.board[action.cell.row-1][action.cell.column-1];
    els.candidates.textContent=`Selected cell candidates: ${names(mask).join(', ')}`;
  }else if(frame.stack.length){
    const b=frame.stack.at(-1);
    els.candidates.textContent=`Current branch: ${b.value} at row ${b.cell.row}, column ${b.cell.column}`;
  }else els.candidates.textContent='';
}
function render(){
  const puzzle=PUZZLES[selected],frame=frames[position],action=position?puzzle.events[position-1]:null;
  els['step-label'].textContent=`Step ${position} / ${puzzle.events.length}`;
  els.step.value=position;
  els.previous.disabled=position===0;
  els.next.disabled=position===puzzle.events.length;
  els['puzzle-title'].textContent=puzzle.id;
  els.board.innerHTML=svgBoard(puzzle,frame.board,action);
  describe(puzzle,frame,action,position);
}
function choosePuzzle(index){
  selected=index;position=0;frames=buildFrames(PUZZLES[selected]);
  els.step.max=PUZZLES[selected].events.length;
  render();
}
function jumpTo(type){
  const events=PUZZLES[selected].events;
  let index=events.findIndex((event,i)=>i>=position&&event.type===type);
  if(index<0)index=events.findIndex(event=>event.type===type);
  if(index>=0){position=index+1;render()}
}
PUZZLES.forEach((puzzle,index)=>{
  const option=document.createElement('option');option.value=index;
  const contradictions=puzzle.events.filter(event=>event.type==='contradiction').length;
  option.textContent=`${puzzle.id} · ${puzzle.events.length} steps${contradictions?' · contradiction':''}`;
  els.puzzle.append(option);
});
els.puzzle.addEventListener('change',()=>choosePuzzle(Number(els.puzzle.value)));
els.step.addEventListener('input',()=>{position=Number(els.step.value);render()});
els.previous.addEventListener('click',()=>{if(position>0){position--;render()}});
els.next.addEventListener('click',()=>{if(position<PUZZLES[selected].events.length){position++;render()}});
els['next-branch'].addEventListener('click',()=>jumpTo('branch'));
els['next-conflict'].addEventListener('click',()=>jumpTo('contradiction'));
document.addEventListener('keydown',event=>{
  if(['SELECT','INPUT','BUTTON'].includes(document.activeElement.tagName))return;
  if(event.key==='ArrowRight'&&position<PUZZLES[selected].events.length){position++;render()}
  if(event.key==='ArrowLeft'&&position>0){position--;render()}
});
choosePuzzle(0);
</script>
</body>
</html>
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    puzzles = compact_traces(args.trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(puzzles, separators=(",", ":")).replace("<", "\\u003c")
    page = PAGE.replace("__PUZZLE_COUNT__", html.escape(str(len(puzzles))))
    page = page.replace("__DATA__", data)
    args.output.write_text(page, encoding="utf-8")
    print(f"{args.output} ({len(puzzles)} puzzles, "
          f"{sum(len(p['events']) for p in puzzles)} actions)")


if __name__ == "__main__":
    main()
