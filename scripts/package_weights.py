#!/usr/bin/env python3
import argparse,hashlib,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
for cohort in ['stad','thca','chol','lihc']:
    files=sorted((ROOT/'checkpoints').glob(f'*_{cohort}_fold*.pt'))
    assert len(files)==100,(cohort,len(files))
    dest=a.output/f'hypermil_weights_{cohort}.tar'
    with tarfile.open(dest,'w') as t:
        for file in files:t.add(file,arcname='checkpoints/'+file.name)
        t.add(ROOT/'checkpoints/manifest.csv',arcname='checkpoints/manifest.csv')
    h=hashlib.sha256()
    with dest.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    dest.with_suffix('.tar.sha256').write_text(h.hexdigest()+'  '+dest.name+'\n')
    print(dest,dest.stat().st_size)
