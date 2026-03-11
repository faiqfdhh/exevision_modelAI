# This script removes all non-breaking spaces from a file and overwrites it in-place.
import sys

if len(sys.argv) != 2:
    print("Usage: python remove_nbsp.py <file>")
    sys.exit(1)

file_path = sys.argv[1]
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all non-breaking spaces (U+00A0) with regular spaces
content = content.replace('\u00A0', ' ')
content = content.replace(chr(160), ' ')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Cleaned: {file_path}")
