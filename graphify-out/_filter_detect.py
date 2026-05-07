import json
from pathlib import Path

with open('graphify-out/.graphify_detect.json', 'r') as f:
    data = json.load(f)

new_files = {}
total_files = 0

cwd = Path('.').resolve()
apps_dir = str(cwd / 'apps')
core_dir = str(cwd / 'core')

for category, files in data.get('files', {}).items():
    filtered_files = [f for f in files if f.startswith(apps_dir) or f.startswith(core_dir)]
    if filtered_files:
        new_files[category] = filtered_files
        total_files += len(filtered_files)

data['files'] = new_files
data['total_files'] = total_files

with open('graphify-out/.graphify_detect.json', 'w') as f:
    json.dump(data, f)

print(f"total_files={total_files}")
for k, v in new_files.items():
    print(f"  {k}: {len(v)} files")
