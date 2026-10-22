#!/usr/bin/env python3
"""Check released file identity and all fresh evaluation reports."""
import csv,json,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hypermil.engine import sha256
index=yaml.safe_load((ROOT/'configs/config_index.yaml').read_text())['methods']
rows=list(csv.DictReader((ROOT/'checkpoints/manifest.csv').open()))
issues=[];evaluated=0;runtime_only_snapshots=[];coverage={};example=[]
for r in rows:
    key=f"{r['method']}/{r['cohort']}/fold{r['fold']}"
    file=ROOT/'checkpoints'/r['filename']
    if sha256(file)!=r['sha256']:issues.append([key,'checkpoint checksum'])
    path=ROOT/'results/eval'/r['method']/r['cohort']/f"fold{r['fold']}"/'report.json'
    if not path.exists():issues.append([key,'missing evaluation']);continue
    report=json.loads(path.read_text());evaluated+=1
    coverage.setdefault((r['method'],r['cohort']),set()).add(int(report['fold']))
    if r['method']=='HyperMIL' and r['cohort']=='STAD':
        example.append({'fold':report['fold'],'cindex':report['cindex']})
    yaml_matches=report['config_sha256']==sha256(ROOT/index[r['method']])
    if not yaml_matches:
        snapshot=ROOT/'docs/config_snapshots'/f"{report['config_sha256']}.yaml"
        if snapshot.exists() and sha256(snapshot)==report['config_sha256']:
            before=yaml.safe_load(snapshot.read_text());after=yaml.safe_load((ROOT/index[r['method']]).read_text())
            before.pop('runtime',None);after.pop('runtime',None)
            yaml_matches=before==after
            if yaml_matches:runtime_only_snapshots.append(key)
    checks={
        'checkpoint':report['checkpoint_sha256']==r['sha256'],
        'engine':report['engine_sha256']==sha256(ROOT/'hypermil/engine.py'),
        'interface':report['interface_sha256']==sha256(ROOT/'hypermil/interfaces.py'),
        'yaml':yaml_matches,
        'strict_load':report['strict_load'],
        'fresh_forward':report['fresh_forward']}
    issues.extend([key,name] for name,ok in checks.items() if not ok)
cohorts=['STAD','THCA','CHOL','LIHC']
complete=all(coverage.get((m,c),set())==set(range(5)) for m in index for c in cohorts)
if not complete:issues.append(['cross_validation','incomplete five-fold coverage'])
report={'status':'PASS' if not issues and len(rows)==400 else 'INCOMPLETE_OR_INVALID',
    'model_files':len(rows),'fresh_evaluated_folds':evaluated,'issues':issues,
    'cross_validation':{'unit':'patient','fold_count':5,'fold_ids':list(range(5)),'cohorts':cohorts,
        'method_count':len(index),'all_method_cohort_pairs_have_five_folds':complete},
    'example':{'method':'HyperMIL','cohort':'STAD','fold_metrics':sorted(example,key=lambda r:r['fold'])},
    'runtime_configuration_notes':{'equivalent_snapshot_count':len(runtime_only_snapshots)},
    'total_checkpoint_bytes':sum(int(r['bytes']) for r in rows),'largest_checkpoint_bytes':max(int(r['bytes']) for r in rows)}
(ROOT/'docs/release_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
if report['status']!='PASS':raise SystemExit(1)
