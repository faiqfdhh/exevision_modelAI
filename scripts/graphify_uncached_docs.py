import json
from pathlib import Path
from graphify.detect import detect
from graphify.cache import check_semantic_cache

p = Path('docs')
if not p.exists():
    print('No docs folder found')
    raise SystemExit(1)

det = detect(p)
all_files = []
for ftype in ('document','paper','image','video','code'):
    all_files.extend(det.get('files', {}).get(ftype, []))

cached_nodes, cached_edges, cached_hyperedges, uncached = check_semantic_cache(all_files)
Path('graphify-out').mkdir(parents=True, exist_ok=True)
Path('graphify-out/.graphify_cached_docs.json').write_text(json.dumps({'nodes': cached_nodes, 'edges': cached_edges, 'hyperedges': cached_hyperedges}, indent=2))
Path('graphify-out/.graphify_uncached_docs.txt').write_text('\n'.join(uncached))
print(f'Checked {len(all_files)} files: {len(all_files)-len(uncached)} cached, {len(uncached)} uncached')
