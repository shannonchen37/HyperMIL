#!/usr/bin/env python3
"""Run all methods through scripts/evaluate.py and summarize computed metrics."""
import argparse,csv,json,os,statistics,subprocess,sys,time,yaml
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def aggregate():
    cohorts=['STAD','THCA','CHOL','LIHC']
    reports={}
    for path in (ROOT/'results/eval').glob('*/*/fold*/report.json'):
        report=json.loads(path.read_text())
        reports[(report['method'],report['cohort'],int(report['fold']))]=report
    fold_rows=[{'method':m,'cohort':c,'fold':f,'patients':r['patients'],'cindex':r['cindex']}
               for (m,c,f),r in sorted(reports.items())]
    rows=[]
    for method in sorted({m for m,c,f in reports}):
        cohort_means={}
        for cohort in cohorts:
            folds=[f for f in range(5) if (method,cohort,f) in reports]
            if not folds:continue
            scores=[reports[(method,cohort,f)]['cindex'] for f in folds]
            mean=statistics.mean(scores)
            sd=statistics.stdev(scores) if len(scores)>1 else ''
            rows.append({'method':method,'cohort':cohort,'fold_count':len(folds),
                         'folds':' '.join(map(str,folds)),'mean':mean,'sample_sd':sd})
            if len(folds)==5:cohort_means[cohort]=mean
        if len(cohort_means)==len(cohorts):
            rows.append({'method':method,'cohort':'Macro','fold_count':'',
                         'folds':'','mean':statistics.mean(cohort_means.values()),'sample_sd':''})
    output=ROOT/'results';output.mkdir(parents=True,exist_ok=True)
    for name,fieldnames,values in [
        ('fold_metrics.csv',['method','cohort','fold','patients','cindex'],fold_rows),
        ('summary_metrics.csv',['method','cohort','fold_count','folds','mean','sample_sd'],rows),
    ]:
        with (output/name).open('w',newline='') as destination:
            writer=csv.DictWriter(destination,fieldnames=fieldnames)
            writer.writeheader();writer.writerows(values)
    summary={'fold_reports':len(fold_rows),'cohort_summaries':sum(r['cohort']!='Macro' for r in rows),
             'complete_five_fold_summaries':sum(r['cohort']!='Macro' and r['fold_count']==5 for r in rows),
             'macro_summaries':sum(r['cohort']=='Macro' for r in rows),
             'fold_metrics':'results/fold_metrics.csv','summary_metrics':'results/summary_metrics.csv'}
    (output/'eval_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for row in rows:
        sd='' if row['sample_sd']=='' else f" ± {row['sample_sd']:.4f}"
        count=f" ({row['fold_count']}/5 folds)" if row['cohort']!='Macro' else ''
        print(f"{row['method']} {row['cohort']}: {row['mean']:.4f}{sd}{count}",flush=True)
    print(json.dumps(summary),flush=True)
    return summary

def run(gpus,smoke=False,selected_configs=None):
    from hypermil.engine import sha256
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
    aggregate()
    if errors:raise SystemExit(json.dumps(errors))

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpus',default='auto');p.add_argument('--config',action='append');p.add_argument('--smoke',action='store_true');p.add_argument('--aggregate',action='store_true')
    a=p.parse_args()
    if a.aggregate:aggregate()
    else:run(None if a.gpus=='auto' else [int(x) for x in a.gpus.split(',')],a.smoke,a.config)

if __name__=='__main__':main()
