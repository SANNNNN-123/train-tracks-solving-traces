import json, statistics as st
from collections import Counter, defaultdict

d=json.load(open('output/eval/run-03-legal-action-menu/medium/rollout.json'))
tids={p['puzzle_id'] for p in d['puzzles']}
teacher=defaultdict(list)
for line in open('data/train_medium.jsonl',encoding='utf-8'):
    r=json.loads(line)
    if r['puzzle_id'] in tids: teacher[r['puzzle_id']].append(r)
for k in teacher: teacher[k].sort(key=lambda r:r['step_index'])
TILES=("XX","NS","EW","NE","NW","SE","SW"); BIT={n:1<<i for i,n in enumerate(TILES)}
def mask(c): return sum(BIT[x] for x in c)

def lost_mask(h):
    stack=[]; owner=[tuple()]*len(h)
    for i,ev in enumerate(h):
        t=ev['action']['type']
        if t=='branch': stack.append(i); owner[i]=tuple(stack)
        elif t=='undo':
            ts=ev['action']['to_step']
            while stack and stack[-1]!=ts: stack.pop()
            if stack: stack.pop()
            owner[i]=tuple(stack)
        else: owner[i]=tuple(stack)
    undone=[]
    # branches that are targeted by an undo
    for ev in h:
        if ev['action']['type']=='undo': undone.append(ev['action']['to_step'])
    lost=[any(b in undone for b in owner[i]) for i in range(len(h))]
    return lost

def replay(p):
    h=p['history']; ref=p['reference']; H=len(ref); W=len(ref[0])
    board=[row[:] for row in h[0]['state']['domains']]; stack=[]
    first_wrong=None; bad=0
    for i,ev in enumerate(h):
        a=ev['action']; t=a['type']
        if t=='deduce':
            r,c=a['cell']['row']-1,a['cell']['column']-1; board[r][c]=mask(a['candidates'])
        elif t=='branch':
            r,c=a['cell']['row']-1,a['cell']['column']-1
            stack.append((a['cell'],a['value'],[row[:] for row in board]))
            if first_wrong is None and a['value']!=ref[r][c]:
                first_wrong=(i,len(ev['state']['branch_stack']))
            board[r][c]=BIT[a['value']]
        elif t=='undo':
            cell,val,cp=stack.pop(); board=[row[:] for row in cp]
            r,c=cell['row']-1,cell['column']-1; board[r][c]&=~BIT[val]
    poison=sum(1 for r in range(H) for c in range(W) if not (board[r][c]&BIT[ref[r][c]]))
    active_wrong=sum(1 for cell,val,cp in stack if val!=ref[cell['row']-1][cell['column']-1])
    return poison, active_wrong, first_wrong

print('| puzzle | shape | solved | model | teacher | lost | useful | wrong | poison@end | first wrong |')
print('|---|---|:--:|--:|--:|--:|--:|--:|--:|---|')
rows=[]
for p in d['puzzles']:
    h=p['history']; lost=sum(lost_mask(h))
    tn=len(teacher[p['puzzle_id']])
    wrong=sum(1 for ev in h if ev['action']['type']=='branch' and ev['action']['value']!=p['reference'][ev['action']['cell']['row']-1][ev['action']['cell']['column']-1])
    shape=f"{len(p['reference'])}x{len(p['reference'][0])}"
    poison,aw,fw=replay(p)
    fw_s=f"step {fw[0]}, depth {fw[1]}" if fw else "-"
    print(f"| {p['puzzle_id']} | {shape} | {'yes' if p['solved'] else 'no'} | {len(h)} | {tn} | {lost} | {len(h)-lost} | {wrong} | {poison} | {fw_s} |")
    rows.append(dict(pid=p['puzzle_id'],solved=p['solved'],shape=shape,n=len(h),tn=tn,lost=lost,useful=len(h)-lost,
                     wrong=wrong,poison=poison,aw=aw,ded=sum(1 for e in h if e['action']['type']=='deduce'),
                     final=sum(1 for r in h[-1]['state']['domains'] for c in r if bin(c).count('1')==1)))

print()
for name,sel in [('SOLVED',lambda s:s),('FAILED',lambda s:not s)]:
    rs=[r for r in rows if sel(r['solved'])]
    print(name,'n',len(rs))
    for k in ['n','tn','lost','useful','wrong','poison','final']:
        v=[r[k] for r in rs]; print(f'  {k:8} mean={st.mean(v):7.1f} median={st.median(v):7.1f}')
    print('  lost/actions mean %.1f%%'%(100*st.mean([r['lost']/r['n'] for r in rs])))
    print('  wrong/branches mean %.1f%%'%(100*st.mean([r['wrong']/max(1,sum(1 for e in [])+r['wrong']) for r in rs])) if False else '')
