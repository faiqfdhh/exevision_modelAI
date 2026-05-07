import json
from graphify.detect import detect
from pathlib import Path
result = detect(Path('.'))
with open('graphify-out/.graphify_detect.json', 'w') as f:
    json.dump(result, f)
d = result
print(f"total_files={d['total_files']}, total_words={d['total_words']}")
for k, v in d.get('files', {}).items():
    if v:
        print(f"  {k}: {len(v)} files")
