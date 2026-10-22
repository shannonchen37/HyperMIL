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
issues=[];evaluated=0;fold_differences=[];runtime_only_snapshots=[]
for r in rows:
    key=f"{r['method']}/{r['cohort']}/fold{r['fold']}"
    file=ROOT/'checkpoints'/r['filename']
    if sha256(file)!=r['sha256']:issues.append([key,'checkpoint checksum'])
    path=ROOT/'results/eval'/r['method']/r['cohort']/f"fold{r['fold']}"/'report.json'
    if not path.exists():issues.append([key,'missing evaluation']);continue
    report=json.loads(path.read_text());evaluated+=1
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
    if r['expected_cindex'] and abs(report['cindex']-float(r['expected_cindex']))>1e-15:
        fold_differences.append({'model':key,'reference':float(r['expected_cindex']),'eval':report['cindex']})
    issues.extend([key,name] for name,ok in checks.items() if not ok)
comparison=list(csv.DictReader((ROOT/'results/paper_comparison.csv').open()))
for r in comparison:
    if r['status']!='PASS':issues.append([f"Table {r['table']} {r['method']} {r['cohort']}",r['status']])
report={'status':'PASS' if not issues and len(rows)==400 and len(comparison)==105 else 'INCOMPLETE_OR_MISMATCH',
    'model_files':len(rows),'fresh_evaluated_folds':evaluated,'table_cells':len(comparison),
    'passed_table_cells':sum(r['status']=='PASS' for r in comparison),'issues':issues,
    'per_fold_reference_differences':fold_differences,
    'equivalent_runtime_only_config_snapshots':runtime_only_snapshots,
    'total_checkpoint_bytes':sum(int(r['bytes']) for r in rows),'largest_checkpoint_bytes':max(int(r['bytes']) for r in rows)}
(ROOT/'docs/release_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
if report['status']!='PASS':raise SystemExit(1)
