#!/usr/bin/env python3
"""Run all methods through scripts/evaluate.py and compare the paper tables."""
import argparse,csv,json,os,statistics,subprocess,sys,time,yaml
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hypermil.engine import sha256

def aggregate():
    aliases=yaml.safe_load((ROOT/'configs/config_index.yaml').read_text()).get('aliases',{})
    targets=list(csv.DictReader((ROOT/'results/paper_targets.csv').open()))
    reports={}
    for p in (ROOT/'results/eval').glob('*/*/fold*/report.json'):
        r=json.loads(p.read_text());reports[(r['method'],r['cohort'],r['fold'])]=r
    rows=[]
    for t in targets:
        method=aliases.get(t['method'],t['method']);cohort=t['cohort']
        cs=['STAD','THCA','CHOL','LIHC'] if cohort=='Macro' else [cohort]
        keys=[(method,c,f) for c in cs for f in range(5)]
        if not all(k in reports for k in keys):
            rows.append({**t,'eval_mean':'','eval_sample_sd':'','status':'PENDING'});continue
        means=[statistics.mean(reports[(method,c,f)]['cindex'] for f in range(5)) for c in cs]
        mean=statistics.mean(means)
        sd='' if cohort=='Macro' else statistics.stdev(reports[k]['cindex'] for k in keys)
        passed=f'{mean:.4f}'==f"{float(t['mean']):.4f}" and (sd=='' or f'{sd:.4f}'==f"{float(t['sample_sd']):.4f}")
        rows.append({**t,'eval_mean':mean,'eval_sample_sd':sd,'status':'PASS' if passed else 'MISMATCH'})
    with (ROOT/'results/paper_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={'fold_reports':len(reports),'table_cells':len(rows),'passed':sum(r['status']=='PASS' for r in rows),
        'mismatches':[r for r in rows if r['status']=='MISMATCH'],'pending':sum(r['status']=='PENDING' for r in rows)}
    (ROOT/'results/verification_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Attach fresh metrics to both public and private mapping without changing weights.
    for path in [ROOT/'checkpoints/manifest.csv',ROOT.parent/'HyperMIL_release_audit_20261022/checkpoint_source_mapping.csv']:
        if not path.exists(): continue
        mapping=list(csv.DictReader(path.open()))
        for r in mapping:
            key=(r['method'],r['cohort'],int(r['fold']))
            if key in reports:
                r['eval_cindex']=reports[key]['cindex'];r['eval_status']='EVALUATED'
            else:r.setdefault('eval_cindex','')
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(mapping[0]));w.writeheader();w.writerows(mapping)
    print(json.dumps(summary),flush=True)
    return summary

def run(gpus,smoke=False,selected_configs=None):
    configs=yaml.safe_load((ROOT/'configs/config_index.yaml').read_text())['methods']
    if selected_configs:configs={m:p for m,p in configs.items() if p in selected_configs}
    gpu_rows=subprocess.run(['nvidia-smi','--query-gpu=index,name','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()
    gpu_names={int(r.split(',')[0]):r.split(',',1)[1].strip() for r in gpu_rows}
    if gpus is None:gpus=list(gpu_names)
    runtime=yaml.safe_load((ROOT/'configs/runtime.yaml').read_text()) if (ROOT/'configs/runtime.yaml').exists() else {}
    preferences={m:runtime.get('preferred_gpu_models',{}).get(p) for m,p in configs.items()}
    methods=list(configs)
    tasks=[(m,c,f) for m in methods for c in (['CHOL'] if smoke else ['STAD','THCA','CHOL','LIHC']) for f in ([0] if smoke else ['all'])]
    active={};errors=[]
    while tasks or active:
        for gpu in gpus:
            if gpu in active or not tasks:continue
            candidate=next((i for i,(m,c,f) in enumerate(tasks) if not preferences[m] or preferences[m]==gpu_names[gpu] or preferences[m] not in [gpu_names[g] for g in gpus]),None)
            if candidate is None:continue
            m,c,f=tasks.pop(candidate);base=ROOT/'results/eval'/m/c
            folder=base if f=='all' else base/f'fold{f}'
            complete=True
            for fold in (range(5) if f=='all' else [f]):
                report=base/f'fold{fold}'/'report.json'
                if not report.exists():complete=False;break
                r=json.loads(report.read_text())
                if r.get('engine_sha256')!=sha256(ROOT/'hypermil/engine.py') or r.get('interface_sha256')!=sha256(ROOT/'hypermil/interfaces.py') or r.get('config_sha256')!=sha256(ROOT/configs[m]):complete=False;break
            if complete:continue
            folder.mkdir(parents=True,exist_ok=True);log=(folder/'run.log').open('w')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/evaluate.py'),'--config',configs[m],'--cohort',c,'--fold',str(f)],env=env,stdout=log,stderr=subprocess.STDOUT)
            active[gpu]=(proc,log,m,c,f)
            print('START',gpu,m,c,f,flush=True)
        for gpu,(proc,log,m,c,f) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            log.close();del active[gpu]
            if code:errors.append({'method':m,'cohort':c,'fold':f,'exit_code':code})
            print('DONE',m,c,f,code,flush=True)
        status={'pending':len(tasks),'active':{str(g):{'pid':p.pid,'method':m,'cohort':c,'fold':f} for g,(p,l,m,c,f) in active.items()},'errors':errors}
        (ROOT/'results/run_status.json').write_text(json.dumps(status,indent=2)+'\n')
        if tasks or active:time.sleep(2)
    summary=aggregate()
    if errors:raise SystemExit(json.dumps(errors))
    if not smoke and not selected_configs and (summary['pending'] or summary['mismatches']):raise SystemExit('Paper comparison has unresolved cells')

p=argparse.ArgumentParser();p.add_argument('--gpus',default='auto');p.add_argument('--config',action='append');p.add_argument('--smoke',action='store_true');p.add_argument('--aggregate',action='store_true')
a=p.parse_args()
if a.aggregate:aggregate()
else:run(None if a.gpus=='auto' else [int(x) for x in a.gpus.split(',')],a.smoke,a.config)
