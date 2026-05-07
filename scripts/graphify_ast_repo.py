import json
from pathlib import Path
from graphify.detect import detect
from graphify.extract import collect_files, extract

root = Path('.')
print('Running detect on repo root...')
d = detect(root)
all_files = []
for ftype, files in d.get('files', {}).items():
    for f in files:
        p = Path(f)
        if p.is_dir():
            all_files.extend(collect_files(p))
        else:
            all_files.append(p)

print(f'Files discovered: {len(all_files)}')
res = {'nodes': [], 'edges': [], 'input_tokens': 0, 'output_tokens': 0}
if all_files:
    print('Extracting AST for discovered files (this may take a moment)...')
    res = extract(all_files)
else:
    print('No files found to extract')

Path('graphify-out').mkdir(parents=True, exist_ok=True)
out_path = Path('graphify-out/.graphify_ast_repo.json')
out_path.write_text(json.dumps(res, indent=2))
print('AST extraction complete. Wrote', out_path)
