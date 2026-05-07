import json
from graphify.detect import detect
from pathlib import Path
r = detect(Path('docs'))
Path('graphify-out/.graphify_detect_docs.json').write_text(json.dumps(r, indent=2))
print('Corpus:', r.get('total_files',0), 'files, ~' + str(r.get('total_words',0)), 'words')
for ftype, files in r.get('files', {}).items():
    if files:
        print('  {}: {} files'.format(ftype, len(files)))
