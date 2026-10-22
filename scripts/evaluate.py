#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hypermil.engine import evaluate
p=argparse.ArgumentParser()
p.add_argument('--config',required=True);p.add_argument('--cohort',required=True,choices=['STAD','THCA','CHOL','LIHC'])
p.add_argument('--fold',default='all',choices=['all','0','1','2','3','4']);p.add_argument('--device',default='cuda:0')
p.add_argument('--checkpoint');p.add_argument('--output')
a=p.parse_args()
if a.fold=='all' and (a.checkpoint or a.output):p.error('Explicit checkpoint/output requires one fold')
for fold in (range(5) if a.fold=='all' else [int(a.fold)]):
    evaluate(a.config,a.cohort,fold,a.device,a.checkpoint,a.output)
