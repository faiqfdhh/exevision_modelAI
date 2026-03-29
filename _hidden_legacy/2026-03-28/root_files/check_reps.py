import json, os
p = r'D:\squat\unlabeled_features\raw_unfiltered\segmented_reps'
total_reps = 0
empty = 0
for f in os.listdir(p):
    if not f.endswith('.json'): continue
    d = json.load(open(os.path.join(p, f)))
    n = len(d.get('repetitions', []))
    total_reps += n
    if n == 0: empty += 1
print(f'Total reps: {total_reps}')
print(f'Videos with 0 reps: {empty}')
print(f'Videos with reps: {5250 - empty}')
print(f'Avg reps per video: {total_reps / 5250:.1f}')
