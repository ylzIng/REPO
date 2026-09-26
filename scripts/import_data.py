"""Import and checksum an existing authorized copy of the prepared datasets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
args = parser.parse_args()
target = Path(__file__).resolve().parents[1] / 'data'
entries = json.loads((target / 'manifest.json').read_text())
# Verify all inputs before copying anything.
for entry in entries:
    source = args.source / entry['path']
    if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != entry['sha256']:
        raise SystemExit(f"Missing or mismatched source: {source}")
for entry in entries:
    destination = target / entry['path']
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (args.source / entry['path']).resolve() != destination.resolve():
        shutil.copy2(args.source / entry['path'], destination)
print(f'[REPO] Imported and verified {len(entries)} parquet files.')
