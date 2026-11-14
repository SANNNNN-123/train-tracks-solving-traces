import json, statistics as st
from collections import Counter, defaultdict

ROLL='output/eval/run-03-legal-action-menu/medium/rollout.json'
d=json.load(open(ROLL))
tids={p['puzzle_id'] for p in d['puzzles']}
teacher=defaultdict(list)
for line in open('data/train_medium.jsonl',encoding='utf-8'):
    r=json.loads(line)
    if r['puzzle_id'] in tids: teacher[r['puzzle_id']].append(r)
for k in teacher: teacher[k].sort(key=lambda r:r['step_index'])

def popcount(x): return bin(x).count('1')

def lost_mask(hist):
    """Mark each action index as belonging to a branch that is eventually undone."""
    n=len(hist)
    # entries: (branch_index, undo_index) structure via stack
    stack=[]  # list of dict(idx, children)
    owner=[None]*n
    for i,a in enumerate(hist):
        t=a['action']['type']
        if t=='branch':
            stack.append(i)
            owner[i]=tuple(stack)
        elif t=='undo':
            if stack: stack.pop()
            owner[i]=tuple(stack)
        else:
            owner[i]=tuple(stack)
    # a branch is undone if there is an 'undo' action whose stack (after pop) excludes it.
    # directly: a branch index b is undone if some undo action at i has owner[i] == tuple stack prefix before b.
    undone=set()
    for i,a in enumerate(hist):
        if a['action']['type']=='undo':
            # the branch being undone is the last branch currently in stack before pop;
            # since owner[i] is after pop, the undone branch has owner tuple == owner[i]+(b,)
            prefix=owner[i]
            # find branch index b with owner[b]==prefix+(b,)
            for j in range(i-1,-1,-1):
                if hist[j]['action']['type']=='branch' and owner[j]==prefix+(j,):
                    undone.add(j); break
    lost=[False]*n
    for i in range(n):
        if any(b in undone for b in owner[i]): lost[i]=True
    return lost, undone

print(f"{'pid':18}{'S':2}{'act':4}{'lost':5}{'prod':5}{'prodDed':7}{'teacher':8}{'prod:teach':11}{'wrongBr':8}{'finalCell':10}")
rows=[]
for p in d['puzzles']:
    h=p['history']; lost,undone=lost_mask(h)
    lost_a=sum(lost)
    prod=len(h)-lost_a
    prod_ded=sum(1 for i,a in enumerate(h) if a['action']['type']=='deduce' and not lost[i])
    tn=len(teacher[p['puzzle_id']])
    wrong=sum(1 for i,a in enumerate(h) if a['action']['type']=='branch' and a['action']['value']!=p['reference'][a['action']['cell']['row']-1][a['action']['cell']['column']-1])
    rows.append((p['puzzle_id'],p['solved'],len(h),lost_a,prod,prod_ded,tn,wrong,sum(popcount(c)==1 for r in h[-1]['state']['domains'] for c in r)))
    print(f"{p['puzzle_id']:18}{'Y' if p['solved'] else 'N':2}{len(h):<4}{lost_a:<5}{prod:<5}{prod_ded:<7}{tn:<8}{prod/tn:<11.2f}{wrong:<8}{rows[-1][-1]:<10}")

print()
for name,sel in [('SOLVED',lambda s:s),('FAILED',lambda s:not s)]:
    rs=[r for r in rows if sel(r[1])]
    print(name,'n',len(rs))
    for k,idx in [('actions',2),('lost',3),('productive',4),('productive_deduce',5),('teacher_n',6),('wrong_branch',7),('final_singletons',8)]:
        v=[r[idx] for r in rs]
        print(f"  {k:20} mean={st.mean(v):6.1f} median={st.median(v):6.1f}")
    print(f"  productive/teacher     mean={st.mean([r[4]/r[6] for r in rs]):.2f}")
    print(f"  lost/actions           mean={st.mean([r[3]/r[2] for r in rs]):.1%}")

# how many failed would have completed under a higher cap if the *waste rate* stayed constant?
# counterfactual: productive actions needed ~ teacher deductions+overhead; estimate
print("\n== counterfactual cap needed at observed efficiency (teacher_n / (productive/actions)) ==")
for r in rows:
    if not r[1]:
        ratio=r[4]/r[2]
        est=r[6]/ratio if ratio else float('inf')
        print(f"  {r[0]:18} waste={1-ratio:.1%} est_cap~{est:.0f} (teacher {r[6]})")

# teacher shrink efficiency
print("\n== teacher vs model productive shrink ==")
def shrink_of(hist,lost=None):
    s=0
    for i,a in enumerate(hist):
        if a['action']['type']=='deduce' and (lost is None or not lost[i]):
            c=a['action']['cell']; before=a['state']['domains'][c['row']-1][c['column']-1]
            after=sum(1<<{'XX':0,'NS':1,'EW':2,'NE':3,'NW':4,'SE':5,'SW':6}[x] for x in a['action']['candidates'])
            s+=popcount(before)-popcount(after)
    return s
for name,sel in [('SOLVED',lambda s:s),('FAILED',lambda s:not s)]:
    ps=[p for p in d['puzzles'] if sel(p['solved'])]
    lost_s=[shrink_of(p['history'],lost_mask(p['history'])[0]) for p in ps]
    all_s=[shrink_of(p['history']) for p in ps]
    t_s=[shrink_of(teacher[p['puzzle_id']]) for p in ps]
    t_d=[sum(1 for a in teacher[p['puzzle_id']] if a['action']['type']=='deduce') for p in ps]
    print(f"{name}: model productive shrink mean={st.mean(lost_s):.0f} (all {st.mean(all_s):.0f}) teacher shrink={st.mean(t_s):.0f} teacher ded={st.mean(t_d):.0f} model prod ded={st.mean([sum(1 for i,a in enumerate(p['history']) if a['action']['type']=='deduce' and not lost_mask(p['history'])[0][i]) for p in ps]):.0f}")
    print(f"   shrink per productive deduce: model={st.mean([ls/pd for ls,pd in zip(lost_s,[sum(1 for i,a in enumerate(p['history']) if a['action']['type']=='deduce' and not lost_mask(p['history'])[0][i]) for p in ps])]):.2f} teacher={st.mean([a/b for a,b in zip(t_s,t_d)]):.2f}")
