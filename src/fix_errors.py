# fix_errors.py
import json
from pathlib import Path

SKIP_PAGES = {4}  # pages to permanently skip (too large / not needed)

f = Path('data/images/vision_descriptions.json')
data = json.load(f.open())

errors = [k for k, v in data.items() if v['description'].startswith('ERROR')]
print(f'Found {len(errors)} error entries: {sorted(int(k) for k in errors)}')

for k in errors:
    page_num = int(k)
    if page_num in SKIP_PAGES:
        # Replace with a placeholder instead of deleting
        # so the analyzer won't retry it
        data[k]['description'] = 'NO_VISUAL_CONTENT'
        print(f'  Page {page_num}: marked as skipped (too large)')
    else:
        # Delete so analyzer will retry
        del data[k]
        print(f'  Page {page_num}: removed → will retry')

f.write_text(json.dumps(data, ensure_ascii=False, indent=2))
print('\nDone. Now run: python src/vision_analyzer.py')