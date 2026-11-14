import json
from collections import Counter, defaultdict
d=json.load(open('output/eval/run-03-legal-action-menu/medium/rollout.json'))
TILES=['XX','NS','EW','NE','NW','SE','SW']
# training positive branch value counts (from data/train.jsonl, excluding dev, computed earlier)
prior_counts={'XX':214,'SE':97,'EW':88,'NS':62,'NE':43,'SW':34,'NW':0}
prior_order=[t for t in sorted(TILES,key=lambda t:-prior_counts[t])]
print('training branch prior order:',prior_order)

def pair_truths(sel):
    stat=defaultdict(Counter)
    rows=[]
    for p in d['puzzles']:
        if not sel(p): continue
        for a in p['history']:
            if a['action']['type']!='branch': continue
            cell=a['action']['cell']; dom=tuple(sorted(t for t in TILES if a['state']['domains'][cell['row']-1][cell['column']-1]&(1<<TILES.index(t))))
            truth=p['reference'][cell['row']-1][cell['column']-1]
            stat[dom][truth]+=1
            rows.append((dom,a['action']['value'],truth))
    return stat,rows

for lbl,sel in [('ALL',lambda p:True),('SOLVED',lambda p:p['solved']),('FAILED',lambda p:not p['solved'])]:
    stat,rows=pair_truths(sel)
    orac=0; model=0; glob=0
    for dom,chosen,truth in rows:
        orac += stat[dom].most_common(1)[0][0]==truth
        model += chosen==truth
        glob += next(t for t in prior_order if t in dom)==truth
    n=len(rows)
    print(f"{lbl}: n={n} model={model/n:.1%} global-training-prior={glob/n:.1%} per-pair-oracle-prior={orac/n:.1%}")
    print('   pair correct-value distributions:', {k:dict(v) for k,v in sorted(stat.items(), key=lambda kv:-sum(kv[1].values()))})
