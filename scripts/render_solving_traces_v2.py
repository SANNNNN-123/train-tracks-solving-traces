"""Render the solving trace viewer with track images from data/assets."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import render_solving_traces as base


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "solving-traces-V2.html"

TILE_IMAGES = (
    "const TILE_IMAGES=Object.fromEntries(['EW','NS','NE','SE','SW','NW']."
    "map(name=>[name,`data/assets/boardtracks-${name}.svg`]));"
)

STRETCHED_BOARD = "#board svg {display:block;width:100%;max-width:700px;margin:auto}"
NATURAL_BOARD = "#board svg {display:block;max-width:100%;height:auto;margin:auto}"
SVG_OPEN = '<svg viewBox="0 0 ${W} ${H}" role="img"'
SIZED_SVG_OPEN = '<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img"'
OLD_LEGEND_COLORS = ".given {background:#245baa}.placed {background:#252b35}.branch {background:#edab34}.changed {background:#8bcfa8}.conflict {background:#e57b78}"
NEW_LEGEND_COLORS = (
    ".given {background:#ccc;border:1px solid #a8b5c4}"
    ".placed {background:linear-gradient(180deg,white 38%,#252b35 38%,#252b35 60%,white 60%);border:1px solid #a8b5c4}"
    ".branch {background:#fff0cd;border:2px solid #b27600}"
    ".changed {background:#ddf3e5;border:2px solid #23834b}"
    ".conflict {background:#fbe2e0;border:2px solid #c63530}"
    ".undo {background:#ece4fa;border:2px solid #7654b0}"
)
OLD_LEGEND_TEXT = """      <span><i class="swatch given"></i> Given rail</span>
      <span><i class="swatch placed"></i> Fixed rail</span>
      <span><i class="swatch changed"></i> Deduction</span>
      <span><i class="swatch branch"></i> Trial branch</span>
      <span><i class="swatch conflict"></i> Contradiction</span>
      <span>× Empty · ?N N candidates</span>"""
NEW_LEGEND_TEXT = """      <span><i class="swatch given"></i> Given track</span>
      <span><i class="swatch placed"></i> Fixed track</span>
      <span><i class="swatch changed"></i> Deduction</span>
      <span><i class="swatch branch"></i> Trial branch</span>
      <span><i class="swatch conflict"></i> Contradiction</span>
      <span><i class="swatch undo"></i> Undo</span>
      <span>× Empty cell</span>
      <span>?N = N remaining choices</span>"""
OLD_COLUMN_CLUES = """  puzzle.columns.forEach((clue,c)=>parts.push(`<text x="${left+(c+.5)*size}" y="29" text-anchor="middle" font-size="19" font-weight="700" fill="#243242">${clue}</text>`));"""
NEW_COLUMN_CLUES = """  puzzle.columns.forEach((clue,c)=>{
    const confirmed=board.reduce((count,line)=>count+(!(line[c]&BITS.XX)?1:0),0);
    parts.push(`<text x="${left+(c+.5)*size}" y="29" text-anchor="middle" font-size="19" font-weight="700" fill="${confirmed===clue?'#008800':'#243242'}">${clue}</text>`);
  });"""
OLD_ROW_CLUES = """  puzzle.rows.forEach((clue,r)=>parts.push(`<text x="${left+w*size+24}" y="${top+(r+.5)*size+6}" text-anchor="middle" font-size="19" font-weight="700" fill="#243242">${clue}</text>`));"""
NEW_ROW_CLUES = """  puzzle.rows.forEach((clue,r)=>{
    const confirmed=board[r].filter(mask=>!(mask&BITS.XX)).length;
    parts.push(`<text x="${left+w*size+24}" y="${top+(r+.5)*size+6}" text-anchor="middle" font-size="19" font-weight="700" fill="${confirmed===clue?'#008800':'#243242'}">${clue}</text>`);
  });"""

VECTOR_TRACKS = """      for(const d of values[0]){
        const [dx,dy]=DIR[d];
        parts.push(`<line x1="${cx}" y1="${cy}" x2="${cx+dx*size/2}" y2="${cy+dy*size/2}" stroke="${color}" stroke-width="4" stroke-linecap="round"/>`);
      }
      if(given.has(cellKey))parts.push(`<circle cx="${cx}" cy="${cy}" r="3" fill="#245baa"/>`);"""

IMAGE_TRACKS = """      parts.push(`<image href="${TILE_IMAGES[values[0]]}" x="${x}" y="${y}" width="${size}" height="${size}" preserveAspectRatio="xMidYMid meet"/>`);
      if(active&&fill!=='white'){
        const border=active.type==='branch'?'#b27600':active.type==='contradiction'?'#c63530':active.type==='undo'?'#7654b0':'#23834b';
        parts.push(`<rect x="${x+2}" y="${y+2}" width="${size-4}" height="${size-4}" fill="none" stroke="${border}" stroke-width="3"/>`);
      }"""

PLAYBACK = """let playbackTimer=null;
function stopPlayback(){
  if(playbackTimer!==null){clearInterval(playbackTimer);playbackTimer=null}
  els.play.textContent='Play';
  els.play.setAttribute('aria-pressed','false');
}
function togglePlayback(){
  if(playbackTimer!==null){stopPlayback();return}
  if(position>=puzzles()[selected].events.length){position=0;render()}
  els.play.textContent='Pause';
  els.play.setAttribute('aria-pressed','true');
  playbackTimer=setInterval(()=>{
    position++;
    render();
    if(position>=puzzles()[selected].events.length)stopPlayback();
  },450);
}
"""


def write_track_assets(directory: Path) -> None:
    """Create transparent board tiles using the reference game's track geometry."""
    directory.mkdir(parents=True, exist_ok=True)
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
            f'y2="{sleeper_end}" stroke="{olive}" stroke-width="{line}"/>'
        )
    for y in (size / 2 - inset, size / 2 + inset):
        straight.append(
            f'<line x1="0" y1="{y}" x2="{size}" y2="{y}" '
            f'stroke="{black}" stroke-width="{line}"/>'
        )

    corner = []
    for index in range(4):
        angle = math.pi / 2 + math.pi / 16 + index * math.pi / 8
        x1 = size + sleeper_start * math.cos(angle)
        y1 = sleeper_start * math.sin(angle)
        x2 = size + sleeper_end * math.cos(angle)
        y2 = sleeper_end * math.sin(angle)
        corner.append(
            f'<line x1="{x1:.4f}" y1="{y1:.4f}" x2="{x2:.4f}" '
            f'y2="{y2:.4f}" stroke="{olive}" stroke-width="{line}"/>'
        )
    for radius in (inset, size - inset):
        points = []
        for index in range(33):
            angle = math.pi / 2 + index * math.pi / 64
            points.append(
                f'{size + radius * math.cos(angle):.4f},'
                f'{radius * math.sin(angle):.4f}'
            )
        corner.append(
            f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="{black}" stroke-width="{line}"/>'
        )

    parts = {
        "EW": "".join(straight),
        "NS": f'<g transform="rotate(90 {size / 2} {size / 2})">{"".join(straight)}</g>',
    }
    for name, angle in (("NE", 0), ("SE", 90), ("SW", 180), ("NW", 270)):
        parts[name] = (
            f'<g transform="rotate({angle} {size / 2} {size / 2})">'
            f'{"".join(corner)}</g>'
        )
    for name, body in parts.items():
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" '
            f'width="{size}" height="{size}">{body}</svg>\n'
        )
        (directory / f"boardtracks-{name}.svg").write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="?", type=Path, default=base.DEFAULT_INPUT)
    parser.add_argument("--medium-trace", type=Path, default=base.DEFAULT_MEDIUM_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    easy_puzzles = base.compact_traces(args.trace)
    medium_puzzles = base.compact_traces(args.medium_trace)
    if base.PAGE.count(VECTOR_TRACKS) != 1:
        raise ValueError("Could not find the vector track renderer")
    if base.PAGE.count(STRETCHED_BOARD) != 1 or base.PAGE.count(SVG_OPEN) != 1:
        raise ValueError("Could not find the original board sizing")
    if base.PAGE.count(OLD_LEGEND_COLORS) != 1 or base.PAGE.count(OLD_LEGEND_TEXT) != 1:
        raise ValueError("Could not find the original legend")
    if base.PAGE.count(OLD_COLUMN_CLUES) != 1 or base.PAGE.count(OLD_ROW_CLUES) != 1:
        raise ValueError("Could not find the original clue renderer")
    page = base.PAGE.replace("const DIR=", TILE_IMAGES + "\nconst DIR=", 1)
    page = page.replace(VECTOR_TRACKS, IMAGE_TRACKS)
    page = page.replace(STRETCHED_BOARD, NATURAL_BOARD)
    page = page.replace(OLD_LEGEND_COLORS, NEW_LEGEND_COLORS)
    page = page.replace(OLD_LEGEND_TEXT, NEW_LEGEND_TEXT)
    page = page.replace(OLD_COLUMN_CLUES, NEW_COLUMN_CLUES)
    page = page.replace(OLD_ROW_CLUES, NEW_ROW_CLUES)
    page = page.replace(SVG_OPEN, SIZED_SVG_OPEN)
    page = page.replace("const size=48,left=22", "const size=45,left=22")
    page = page.replace("    let fill='white';", "    let fill=given.has(cellKey)?'#ccc':'white';")
    page = page.replace("      const color=given.has(cellKey)?'#245baa':'#252b35';\n", "")
    page = page.replace(
        '  <span id="step-label"></span>',
        '  <button id="play" type="button" aria-pressed="false">Play</button>\n'
        '  <span id="step-label"></span>',
    )
    page = page.replace(
        "['mode','puzzle','step','step-label'",
        "['mode','puzzle','step','play','step-label'",
    )
    page = page.replace("function choosePuzzle(index){", PLAYBACK + "function choosePuzzle(index){")
    page = page.replace("function choosePuzzle(index){\n  selected=index;", "function choosePuzzle(index){\n  stopPlayback();\n  selected=index;")
    page = page.replace("function jumpTo(type){\n", "function jumpTo(type){\n  stopPlayback();\n")
    page = page.replace("els.step.addEventListener('input',()=>{position=", "els.step.addEventListener('input',()=>{stopPlayback();position=")
    page = page.replace("els.previous.addEventListener('click',()=>{if(position", "els.previous.addEventListener('click',()=>{stopPlayback();if(position")
    page = page.replace("els.next.addEventListener('click',()=>{if(position", "els.next.addEventListener('click',()=>{stopPlayback();if(position")
    page = page.replace(
        "els['next-branch'].addEventListener",
        "els.play.addEventListener('click',togglePlayback);\n"
        "els['next-branch'].addEventListener",
    )
    page = page.replace("__EASY_COUNT__", html.escape(str(len(easy_puzzles))))
    page = page.replace("__MEDIUM_COUNT__", html.escape(str(len(medium_puzzles))))
    page = page.replace("__EASY_DATA__", json.dumps(easy_puzzles, separators=(",", ":")).replace("<", "\\u003c"))
    page = page.replace("__MEDIUM_DATA__", json.dumps(medium_puzzles, separators=(",", ":")).replace("<", "\\u003c"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_track_assets(args.output.parent / "data" / "assets")
    args.output.write_text(page, encoding="utf-8")
    print(f"{args.output} (easy: {len(easy_puzzles)}, medium: {len(medium_puzzles)})")


if __name__ == "__main__":
    main()
