import json, os
p = r'D:\squat\unlabeled_features\raw_unfiltered\segmented_reps'
multi = 0
for f in os.listdir(p):
    if not f.endswith('.json'): continue
    d = json.load(open(os.path.join(p, f)))
    if len(d.get('repetitions', [])) >= 2: multi += 1
print(f'Videos with 2+ reps: {multi}')
