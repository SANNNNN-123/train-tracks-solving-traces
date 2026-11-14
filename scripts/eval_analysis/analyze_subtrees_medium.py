import json, statistics as st
d=json.load(open('output/eval/run-03-legal-action-menu/medium/rollout.json'))

def branches(p):
    out=[]; stack=[]
    for i,a in enumerate(p['history']):
        t=a['action']['type']
        if t=='branch':
            cell=a['action']['cell']; truth=p['reference'][cell['row']-1][cell['column']-1]
            out.append(dict(i=i,ok=a['action']['value']==truth,undo=None)); stack.append(len(out)-1)
        elif t=='undo':
            if stack:
                bi=stack.pop(); out[bi]['undo']=i
    for b in out:
        if b['undo'] is not None:
            b['size']=b['undo']-b['i']+1
    return out

hdr=('pid','S','wrongBr','lost','lost/wrong','avgWrongSubtree','1stBrOk')
print('%-18s%-2s%-8s%-5s%-11s%-16s%-8s'%hdr)
allsub=[]; lost_by={'SOLVED':[], 'FAILED':[]}
for p in d['puzzles']:
    bs=branches(p)
    wrong=[b for b in bs if not b['ok']]
    undone=[b for b in bs if b['undo'] is not None]
    tops=[b for b in undone if not any(o['i']<b['i']<o['undo'] for o in undone)]
    lost=sum(b['size'] for b in tops)
    subs=[b['size'] for b in wrong if b['undo'] is not None]
    allsub+=subs
    lost_by['SOLVED' if p['solved'] else 'FAILED'].append((lost,len(wrong)))
    first=bs[0]['ok'] if bs else None
    print('%-18s%-2s%-8d%-5d%-11.1f%-16.1f%-8s'%(p['puzzle_id'],'Y' if p['solved'] else 'N',len(wrong),lost,(lost/len(wrong) if wrong else 0),(st.mean(subs) if subs else 0),first))
print()
print('mean size of an undone(wrong) branch subtree:',round(st.mean(allsub),1),'median',st.median(allsub))
print('top-level lost actions: solved mean %.1f, failed mean %.1f'%(st.mean([l for l,w in lost_by['SOLVED']]),st.mean([l for l,w in lost_by['FAILED']])))
print('wrong branches: solved mean %.1f, failed mean %.1f'%(st.mean([w for l,w in lost_by['SOLVED']]),st.mean([w for l,w in lost_by['FAILED']])))
