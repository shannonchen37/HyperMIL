#!/usr/bin/env python3
"""Common configurable training launcher. Eval reproduction uses the released weights."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hypermil.training import train
p=argparse.ArgumentParser()
p.add_argument('--config',required=True);p.add_argument('--cohort',required=True)
p.add_argument('--fold',required=True,type=int,choices=range(5));p.add_argument('--device',default='cuda:0')
p.add_argument('--output',required=True)
a=p.parse_args();train(a.config,a.cohort,a.fold,a.device,a.output)
