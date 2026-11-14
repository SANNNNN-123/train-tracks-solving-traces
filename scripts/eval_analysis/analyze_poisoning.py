import json
from collections import Counter
d=json.load(open('output/eval/run-03-legal-action-menu/medium/rollout.json'))
TILES=("XX","NS","EW","NE","NW","SE","SW")
BIT={n:1<<i for i,n in enumerate(TILES)}

def mask(cands): return sum(BIT[c] for c in cands)

print('pid               solved act firstPoison finalPoison badUndo actionsAfterPoison')
summary=[]
for p in d['puzzles']:
    h=p['history']; ref=p['reference']
    H=len(ref); W=len(ref[0])
    board=[row[:] for row in h[0]['state']['domains']]
    stack=[]
    # initial check
    def poisoned_cells(b):
        return [(r,c) for r in range(H) for c in range(W) if not (b[r][c] & BIT[ref[r][c]])]
    first=None; bad=0; after=0
    for i,ev in enumerate(h):
        a=ev['action']; t=a['type']
        if t=='deduce':
            r,c=a['cell']['row']-1,a['cell']['column']-1
            board[r][c]=mask(a['candidates'])
        elif t=='branch':
            r,c=a['cell']['row']-1,a['cell']['column']-1
            stack.append((a['cell'],a['value'],[row[:] for row in board]))
            board[r][c]=BIT[a['value']]
        elif t=='undo':
            cell,val,checkpoint=stack.pop()
            board=[row[:] for row in checkpoint]
            r,c=cell['row']-1,cell['column']-1
            if val==ref[r][c]:
                bad+=1
            board[r][c] &= ~BIT[val]
        # after action
        pc=poisoned_cells(board)
        if pc and first is None: first=i+1
        if first is not None and i+1>first: after+=1
    final=len(poisoned_cells(board))
    print(f"{p['puzzle_id']:18}{str(p['solved']):7}{len(h):4}{str(first):12}{final:12}{bad:8}{after:18}")
    summary.append((p['solved'],first,final,bad,after,len(h)))
print()
for lbl,sel in [('SOLVED',lambda s:s),('FAILED',lambda s:not s)]:
    rs=[r for r in summary if sel(r[0])]
    print(lbl,'n',len(rs))
    print('  poisoned at some point:',sum(1 for r in rs if r[1] is not None),'/',len(rs))
    print('  poisoned at end:',sum(1 for r in rs if r[2]>0),'/',len(rs))
    print('  mean bad undos:',sum(r[3] for r in rs)/len(rs))
    print('  mean actions after poison:',sum(r[4] for r in rs)/len(rs))
