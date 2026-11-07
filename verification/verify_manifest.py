#!/usr/bin/env python3
"""Check release SHA-256 identities without modifying the record."""
import argparse,csv,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));a=p.parse_args();root=a.root.resolve()
rows=list(csv.DictReader((root/'SHA256SUMS.csv').open()));fail=[]
for r in rows:
    path=(root/r['path']).resolve()
    if not path.is_relative_to(root):raise ValueError('Unsafe manifest path')
    h=hashlib.sha256()
    if not path.is_file():fail.append(r['path']);continue
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    if h.hexdigest()!=r['sha256'] or path.stat().st_size!=int(r['bytes']):fail.append(r['path'])
if fail:raise SystemExit('FAILED: '+str(fail))
print(f'PASS: {len(rows)} release files matched their SHA-256 identities.')
