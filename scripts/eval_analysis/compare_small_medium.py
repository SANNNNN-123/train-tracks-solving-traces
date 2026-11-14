import json, statistics as st
from collections import Counter, defaultdict

def load(path):
    return json.load(open(path))

def popcount(x): return bin(x).count('1')

def branch_stats(p):
    out=[]; stack=[]
    for i,a in enumerate(p['history']):
        t=a['action']['type']
        if t=='branch':
            cell=a['action']['cell']; truth=p['reference'][cell['row']-1][cell['column']-1]
            out.append(dict(i=i,ok=a['action']['value']==truth,undo=None)); stack.append(len(out)-1)
        elif t=='undo':
            ts=a['action'].get('to_step')
            for b in out:
                if b['i']==ts: b['undo']=i
    return out

def sumstats(ps,label):
    wrong=0; tot=0; lost=0; prod=0; acts=0
    for p in ps:
        h=p['history']; bs=branch_stats(p)
        for b in bs:
            tot+=1; wrong+= (not b['ok'])
        undone=[b for b in bs if b['undo'] is not None]
        # top-level undone = no undone ancestor
        tops=[b for b in undone if not any(o['i']<b['i']<o['undo'] for o in undone)]
        lost+=sum(b['undo']-b['i']+1 for b in tops)
        acts+=len(h)
    print(f"{label}: puzzles={len(ps)} branches={tot} wrong={wrong} ({wrong/tot*100:.1f}%) actions={acts} lost={lost} ({lost/acts*100:.1f}%)")

for path,label in [
    ('output/eval/run-03-legal-action-menu/rollout.json','SMALL run03'),
    ('output/eval/run-03-legal-action-menu/medium/rollout.json','MEDIUM run03'),
]:
    d=load(path)
    ps=d['puzzles']
    print('\n==',label,'==')
    print('keys',list(d.keys()))
    sumstats(ps,'all')
    sumstats([p for p in ps if p['solved']],'solved')
    sumstats([p for p in ps if not p['solved']],'failed')
    # branch choice index bias
    idx=Counter(); vals=Counter()
    for p in ps:
        for a in p['history']:
            if a['action']['type']=='branch':
                idx[json.loads(a['raw'])['choice']]+=1
                vals[a['action']['value']]+=1
    print('branch choice index:',dict(idx))
    print('branch values:',dict(vals))
    print('shapes:',Counter(f"{len(p['reference'])}x{len(p['reference'][0])}" for p in ps))
    print('failure reasons:',Counter(p.get('failure') for p in ps))
