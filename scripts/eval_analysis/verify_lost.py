import json, statistics as st
d=json.load(open('output/eval/run-03-legal-action-menu/medium/rollout.json'))
print('pid               lost_union lost_old diff  intervals')
tot_old=0; tot_new=0
subs=[]
for p in d['puzzles']:
    h=p['history']; n=len(h)
    marks=[False]*n
    for i,ev in enumerate(h):
        if ev['action']['type']=='undo':
            t=ev['action']['to_step']
            for j in range(t,i+1): marks[j]=True
            subs.append(i-t+1)
    lost=sum(marks)
    # old from previous script approach: count deduce lost etc not needed
    tot_new+=lost
    print(f"{p['puzzle_id']:18}{lost:10}{'':9}{'':5}{sum(1 for e in h if e['action']['type']=='undo')}")
print('total lost new',tot_new)
print('mean subtree (all undos) %.1f median %.1f'%(st.mean(subs),st.median(subs)))
print('total undo intervals',len(subs))
