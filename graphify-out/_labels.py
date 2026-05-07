import json
from pathlib import Path

analysis = json.loads(Path('graphify-out/.graphify_analysis.json').read_text())
labels = {}
for cid, nodes in analysis['communities'].items():
    if nodes:
        labels[cid] = ", ".join(nodes[:2]).split('_')[-1]
    else:
        labels[cid] = f"Community {cid}"

Path('graphify-out/.graphify_labels.json').write_text(json.dumps(labels))
print(f"Generated labels for {len(labels)} communities")
