import json
from pathlib import Path
from graphify.detect import detect
from graphify.extract import collect_files, extract

p = Path('docs')
if not p.exists():
    print('No docs folder found at ./docs')
    raise SystemExit(1)

d = detect(p)
code_files = []
for f in d.get('files', {}).get('code', []):
    fp = Path(f)
    if fp.is_dir():
        code_files.extend(collect_files(fp))
    else:
        code_files.append(fp)

out = {'nodes': [], 'edges': [], 'input_tokens': 0, 'output_tokens': 0}
if code_files:
    print(f'Found {len(code_files)} code files under docs; running AST extract...')
    res = extract(code_files)
    out = res
else:
    print('No code files under docs; writing empty AST output')

Path('graphify-out').mkdir(parents=True, exist_ok=True)
Path('graphify-out/.graphify_ast_docs.json').write_text(json.dumps(out, indent=2))
print(f"AST written: {len(out.get('nodes',[]))} nodes, {len(out.get('edges',[]))} edges")
