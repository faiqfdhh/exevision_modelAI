import json
from pathlib import Path
import re

analysis = json.loads(Path('graphify-out/.graphify_analysis.json').read_text())
labels = {}

def get_good_name(nodes):
    for node in nodes:
        # Ignore rationales
        if "rationale" in node:
            continue
        # Check if it's a file path
        if node.endswith("_py"):
            parts = node.split('_')
            # Remove core, exevision, py
            clean_parts = [p for p in parts if p not in ('core', 'exevision', 'py', 'stages', 'training', 'apps', 'desktop', 'ui', 'app')]
            if clean_parts:
                return " ".join(clean_parts).title()
        
    # If no file path found, look for class names or functions
    for node in nodes:
        if "rationale" in node:
            continue
        parts = node.split('_')
        # return the last part, Title case
        if len(parts) > 1:
            # find first part that isn't module path
            return " ".join(parts[-2:]).title()
            
    return "Unknown Concepts"

for cid, nodes in analysis['communities'].items():
    if nodes:
        nodes_str = " ".join(nodes[:10]).lower()
        if "bilstm" in nodes_str and "training" in nodes_str:
            name = "BiLSTM Training"
        elif "stgcn" in nodes_str and "training" in nodes_str:
            name = "ST-GCN Training"
        elif "feedback" in nodes_str:
            name = "Feedback Engine"
        elif "annotation" in nodes_str:
            name = "Annotation Tool UI"
        elif "segmentation" in nodes_str:
            name = "Temporal Segmentation"
        elif "scoring" in nodes_str:
            name = "AQA Scoring"
        elif "extract" in nodes_str:
            name = "Pose Extraction"
        elif "neural" in nodes_str:
            name = "Neural Models & Utils"
        elif "api" in nodes_str or "fastapi" in nodes_str:
            name = "FastAPI Backend"
        elif "legacy" in nodes_str:
            name = "Legacy Implementations"
        else:
            name = get_good_name(nodes)
            if "rationale" in name.lower() or name == "Unknown Concepts":
                 name = "Code Rationale / Patterns"
        
        labels[cid] = name[:30] # Limit length
    else:
        labels[cid] = f"Community {cid}"

Path('graphify-out/.graphify_labels.json').write_text(json.dumps(labels))
print(f"Generated SMART labels for {len(labels)} communities")
