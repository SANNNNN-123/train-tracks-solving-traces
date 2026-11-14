import json
d=json.load(open('output/eval/run-03-legal-action-menu/medium/rollout.json'))
print('pid               firstWrong step depth cell chosen truth stillActive actionsAfter')
for p in d['puzzles']:
    if p['solved']: continue
    h=p['history']; ref=p['reference']
    first=None
    stack=[]
    for i,ev in enumerate(h):
        a=ev['action']; t=a['type']
        if t=='branch':
            cell=a['cell']; truth=ref[cell['row']-1][cell['column']-1]
            stack.append((a['cell'],a['value']))
            if first is None and a['value']!=truth:
                first=dict(step=i,depth=len(ev['state']['branch_stack']),cell=(cell['row'],cell['column']),chosen=a['value'],truth=truth,stack_idx=len(stack)-1)
        elif t=='undo':
            ts=a['to_step']
            # pop until target step
            while stack:
                stack.pop()
                if ts==i:  # not meaningful; use branch step lookup below
                    break
    # recompute still active by replaying to_step targets
    active=[]; actions=[]
    for i,ev in enumerate(h):
        a=ev['action']; t=a['type']
        if t=='branch': active.append((i,a['cell'],a['value']))
        elif t=='undo':
            active=[b for b in active if b[0]!=a['to_step']]
    if first:
        still=any(b[0]==first['step'] for b in active)
        print(f"{p['puzzle_id']:18}{str(first['step']):9}{first['depth']:6}{str(first['cell']):10}{first['chosen']:7}{first['truth']:6}{str(still):12}{len(h)-first['step']:13}")
