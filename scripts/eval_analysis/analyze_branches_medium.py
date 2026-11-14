import json, statistics as st
from collections import Counter, defaultdict

ROLL='output/eval/run-03-legal-action-menu/medium/rollout.json'
d=json.load(open(ROLL))
ref_by={p['puzzle_id']:p['reference'] for p in d['puzzles']}
tids=set(ref_by)

# teacher traces, indexed by step
teacher=defaultdict(list)
for line in open('data/train_medium.jsonl',encoding='utf-8'):
    r=json.loads(line)
    if r['puzzle_id'] in tids:
        teacher[r['puzzle_id']].append(r)
for k in teacher: teacher[k].sort(key=lambda r:r['step_index'])

def branch_value(ac):
    if ac['type']!='branch': return None
    if 'value' in ac: return ac['value']
    c=ac.get('candidates')
    return c[0] if c else None

def analyze(hist, ref, label):
    # replay branch stack to know which branch each action belongs to
    # history entries have state BEFORE action.
    branches=[]  # each: step, cell, val, correct, close_step, close_kind, deductions_inside
    plan=[]  # stack of indices
    ded_before=0
    for a in hist:
        t=a['action']['type']
        st_=a['state']
        if t=='deduce':
            if plan: branches[plan[-1]]['ded_in']+=1
        elif t=='branch':
            ac=a['action']; c=ac['cell']; val=branch_value(ac)
            truth=ref[c['row']-1][c['column']-1]
            br=dict(step=a.get('step'), cell=(c['row'],c['column']), val=val, truth=truth,
                    correct=val==truth, ded_in=0, close=None, depth=len(st_['branch_stack']))
            branches.append(br); plan.append(len(branches)-1)
        elif t=='undo':
            if plan:
                idx=plan.pop(); branches[idx]['close']='undo'
        elif t=='contradiction':
            if plan: branches[plan[-1]]['close']='contradiction'
        elif t=='finish':
            pass
    for i in plan: branches[i]['close']='active-at-cap'
    return branches

print("== model branch choices vs reference ==")
print(f"{'pid':18}{'S':2}{'#br':4}{'onSol':6}{'wrong':6}{'active':7}{'undo':5}{'ded_in':7}")
allb=[]; solved_b=[]; failed_b=[]
for p in d['puzzles']:
    bs=analyze(p['history'], p['reference'], p['puzzle_id'])
    allb+=bs
    (solved_b if p['solved'] else failed_b).extend(bs)
    onsol=sum(b['correct'] for b in bs)
    active=sum(1 for b in bs if b['close']=='active-at-cap')
    undo=sum(1 for b in bs if b['close']=='undo')
    dedin=sum(b['ded_in'] for b in bs)
    print(f"{p['puzzle_id']:18}{'Y' if p['solved'] else 'N':2}{len(bs):<4}{onsol:<6}{len(bs)-onsol:<6}{active:<7}{undo:<5}{dedin:<7}")

for name,bs in [('ALL',allb),('SOLVED',solved_b),('FAILED',failed_b)]:
    if not bs: continue
    print(f"\n{name}: branches={len(bs)} on-solution={sum(b['correct'] for b in bs)} ({sum(b['correct'] for b in bs)/len(bs)*100:.1f}%)")
    print("  close reasons:",Counter(b['close'] for b in bs))
    print("  wrong branches:",sum(1 for b in bs if not b['correct']),"/",len(bs))

# teacher branch stats
print("\n== teacher branch choices ==")
tb_all=[]; tb_sol=[]; tb_fail=[]
for p in d['puzzles']:
    bs=analyze(teacher[p['puzzle_id']], p['reference'], p['puzzle_id'])
    tb_all+=bs; (tb_sol if p['solved'] else tb_fail).extend(bs)
for name,bs in [('ALL',tb_all),('SOLVED',tb_sol),('FAILED',tb_fail)]:
    n=len(bs); ok=sum(b['correct'] for b in bs)
    print(f"{name}: branches={n} on-solution={ok} ({ok/n*100:.1f}%)  close={Counter(b['close'] for b in bs)}")

print("\n== deductions spent inside branches that were undone ==")
tot_ded=0; lost_ded=0; lost_by_pid=[]
for p in d['puzzles']:
    bs=analyze(p['history'],p['reference'],p['puzzle_id'])
    ded=sum(1 for a in p['history'] if a['action']['type']=='deduce')
    lost=sum(b['ded_in'] for b in bs if b['close']=='undo')
    tot_ded+=ded; lost_ded+=lost
    lost_by_pid.append((p['puzzle_id'],p['solved'],ded,lost,lost/ded*100 if ded else 0))
for pid,solved,ded,lost,pct in lost_by_pid:
    print(f"  {pid:18} {'Y' if solved else 'N'} ded={ded:3} lost_to_undo={lost:3} ({pct:4.1f}%)")
print(f"TOTAL ded={tot_ded} lost={lost_ded} ({lost_ded/tot_ded*100:.1f}%)")
