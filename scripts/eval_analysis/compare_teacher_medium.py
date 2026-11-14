import json, statistics as st
from collections import Counter, defaultdict

ROLL='output/eval/run-03-legal-action-menu/medium/rollout.json'
d=json.load(open(ROLL))
tids={p['puzzle_id'] for p in d['puzzles']}

# teacher traces for the same puzzles
teacher=defaultdict(list)
for line in open('data/train_medium.jsonl',encoding='utf-8'):
    r=json.loads(line)
    if r['puzzle_id'] in tids:
        teacher[r['puzzle_id']].append(r)

def teacher_stats(rows):
    rows=sorted(rows,key=lambda r:r['step_index'])
    c=Counter(r['action']['type'] for r in rows)
    depths=[len(r['state']['branch_stack']) for r in rows]
    # branch rows and whether labelled positive
    br=[r for r in rows if r['action']['type']=='branch']
    return dict(n=len(rows), deduce=c['deduce'], branch=c['branch'], contra=c['contradiction'], undo=c['undo'],
                maxdepth=max(depths) if depths else 0, wrong_branch=sum(1 for r in br if not r.get('expert_policy_label',True)))

print("TEACHER traces for the 32 evaluated puzzles")
print(f"{'pid':18}{'n':5}{'ded':5}{'br':4}{'con':4}{'und':4}{'dep':4}{'wrongbr':8}")
rows=[]
for p in d['puzzles']:
    t=teacher[p['puzzle_id']]; ts=teacher_stats(t)
    rows.append((p['puzzle_id'],p['solved'],p['actions'],ts))
    print(f"{p['puzzle_id']:18}{ts['n']:<5}{ts['deduce']:<5}{ts['branch']:<4}{ts['contra']:<4}{ts['undo']:<4}{ts['maxdepth']:<4}{ts['wrong_branch']:<8}")

print()
for label,pred in [('solved',lambda s:s),('failed',lambda s:not s)]:
    sel=[r for r in rows if pred(r[1])]
    print(label,'n',len(sel))
    for k in ['n','deduce','branch','contra','undo','maxdepth','wrong_branch']:
        v=[r[3][k] for r in sel]
        print(f"  teacher {k:10} mean={st.mean(v):6.1f} median={st.median(v):6.1f} min={min(v):4} max={max(v):4}")

print()
print("== teacher length vs model length (failed) ==")
for pid,solved,acts,ts in rows:
    if not solved:
        print(f"{pid:18} model=256 teacher={ts['n']:3}  gap={256-ts['n']:+4}")
print()
print("== teacher length vs model length (solved) ==")
for pid,solved,acts,ts in rows:
    if solved:
        print(f"{pid:18} model={acts:3} teacher={ts['n']:3}  ratio={acts/ts['n']:.2f}")

# branch correctness vs reference
print()
print("== model branch correctness vs reference ==")
tot=0; correct=0
per=[]
for p in d['puzzles']:
    ref=p['reference']
    brs=[]; ok=0
    for a in p['history']:
        ac=a['action']
        if ac['type']=='branch':
            c=ac['cell']; chosen=ac['candidates'][0] if ac.get('candidates') else ac.get('candidate')
            truth=ref[c['row']-1][c['column']-1]
            brs.append((chosen,truth,chosen==truth)); ok+= chosen==truth
    tot+=len(brs); correct+=ok
    if brs:
        per.append((p['puzzle_id'],p['solved'],len(brs),ok,brs))
print(f"total branch actions={tot} correct-vs-reference={correct} ({correct/tot*100:.1f}%)")
for pid,solved,n,ok,brs in per:
    print(f"  {pid:18} solved={solved!s:5} branches={n:2} on-reference={ok:2}")
