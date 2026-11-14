import json, statistics as st
from collections import Counter

P = 'output/eval/run-03-legal-action-menu/medium/rollout.json'
d = json.load(open(P))

def popcount(x): return bin(x).count('1')

rows = []
for p in d['puzzles']:
    h = p['history']
    types = Counter(a['action']['type'] for a in h)
    maxdepth = max((len(a['state']['branch_stack']) for a in h), default=0)
    final = h[-1]['state']['domains'] if h else []
    def singletons(dom): return sum(1 for r in dom for c in r if popcount(c) == 1)
    # progress curve: singletons at every 25 steps
    curve = [singletons(h[i]['state']['domains']) for i in range(0, len(h), 25)]
    # deduces that changed nothing? compare cell before/after
    noops = 0
    shrink_total = 0
    for a in h:
        if a['action']['type'] == 'deduce':
            cell = a['action']['cell']
            # state is BEFORE action
            before = a['state']['domains'][cell['row']-1][cell['column']-1]
            after_mask = sum(1 << {'XX':0,'NS':1,'EW':2,'NE':3,'NW':4,'SE':5,'SW':6}[c] for c in a['action']['candidates'])
            if before == after_mask:
                noops += 1
            shrink_total += popcount(before) - popcount(after_mask)
    rows.append(dict(
        pid=p['puzzle_id'], shape=f"{len(p['reference'])}x{len(p['reference'][0])}",
        solved=p['solved'], actions=len(h), correct=p['correct_cells'], total=p['total_cells'],
        deduce=types['deduce'], branch=types['branch'], contra=types['contradiction'],
        undo=types['undo'], finish=types['finish'], maxdepth=maxdepth,
        singletons=singletons(final), noop_deduce=noops, shrink=shrink_total,
        curve=curve, mean_menu=round(st.mean(a['legal_action_count'] for a in h),1),
        first_branch=next((i for i,a in enumerate(h) if a['action']['type']=='branch'), None),
    ))

print(f"{'pid':18}{'shape':6}{'S':2}{'act':4}{'ded':4}{'br':3}{'con':3}{'und':3}{'dep':4}{'sing':5}{'noop':5}{'shrink':7}{'menu':6}{'1stbr':6}{'corr':7}")
for r in sorted(rows, key=lambda r:(r['solved'], r['shape'])):
    print(f"{r['pid']:18}{r['shape']:6}{'Y' if r['solved'] else 'N':2}{r['actions']:<4}{r['deduce']:<4}{r['branch']:<3}{r['contra']:<3}{r['undo']:<3}{r['maxdepth']:<4}{r['singletons']:<5}{r['noop_deduce']:<5}{r['shrink']:<7}{r['mean_menu']:<6}{str(r['first_branch']):<6}{r['correct']}/{r['total']}")

print()
print("== aggregates ==")
for grp in ['solved','failed']:
    rs=[r for r in rows if (r['solved'] if grp=='solved' else not r['solved'])]
    if not rs: continue
    print(grp, 'n=',len(rs))
    for k in ['actions','deduce','branch','contra','undo','maxdepth','singletons','noop_deduce','shrink','mean_menu']:
        vals=[r[k] for r in rs]
        print(f"  {k:14} mean={st.mean(vals):7.1f} median={st.median(vals):7.1f} min={min(vals):5} max={max(vals):5}")

print()
print("== shapes solved ==")
from collections import defaultdict
by=defaultdict(lambda:[0,0,0,0])
for r in rows:
    b=by[r['shape']]; b[0]+=1; b[1]+=r['solved']; b[2]+=r['correct']; b[3]+=r['total']
for k,v in sorted(by.items()): print(k, f"{v[1]}/{v[0]} cells {v[2]}/{v[3]}")

print()
print("== failed progress curves (singletons per 25 steps) ==")
for r in rows:
    if not r['solved']:
        print(f"{r['pid']:18} {r['shape']:6}", r['curve'])
