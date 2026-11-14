import json
from collections import Counter
d=json.load(open('output/eval/run-03-legal-action-menu/medium/rollout.json'))
print('pid               solved contras distinctContra repeatDetail')
for p in d['puzzles']:
    cs=[(a['action']['anchor'].get('row'),a['action']['anchor'].get('column')) for a in p['history'] if a['action']['type']=='contradiction']
    c=Counter(cs); rep={k:v for k,v in c.items() if v>1}
    if cs:
        print(f"{p['puzzle_id']:18}{str(p['solved']):7}{len(cs):8}{len(c):15} {rep}")
