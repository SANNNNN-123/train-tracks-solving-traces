import json
from collections import Counter
d=json.load(open('output/eval/run-03-legal-action-menu/medium/rollout.json'))
TILES=("XX","NS","EW","NE","NW","SE","SW")
BIT={n:1<<i for i,n in enumerate(TILES)}
def mask(c): return sum(BIT[x] for x in c)

print('pid               solved act finalPoison activeWrongBranch committedPoison stackDepth')
for p in d['puzzles']:
    h=p['history']; ref=p['reference']; H=len(ref); W=len(ref[0])
    board=[row[:] for row in h[0]['state']['domains']]; stack=[]
    for ev in h:
        a=ev['action']; t=a['type']
        if t=='deduce':
            r,c=a['cell']['row']-1,a['cell']['column']-1; board[r][c]=mask(a['candidates'])
        elif t=='branch':
            r,c=a['cell']['row']-1,a['cell']['column']-1
            stack.append((a['cell'],a['value'],[row[:] for row in board])); board[r][c]=BIT[a['value']]
        elif t=='undo':
            cell,val,cp=stack.pop(); board=[row[:] for row in cp]
            r,c=cell['row']-1,cell['column']-1; board[r][c]&=~BIT[val]
    poison=[(r,c) for r in range(H) for c in range(W) if not (board[r][c]&BIT[ref[r][c]])]
    branch_cells={(cell['row']-1,cell['column']-1) for cell,val,cp in stack}
    active_wrong=sum(1 for cell,val,cp in stack if val!=ref[cell['row']-1][cell['column']-1])
    committed=[rc for rc in poison if rc not in branch_cells]
    print(f"{p['puzzle_id']:18}{str(p['solved']):7}{len(h):4}{len(poison):12}{active_wrong:18}{len(committed):16}{len(stack):11}")
