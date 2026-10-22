"""Single evaluation path for every method and ablation."""
import csv, hashlib, json
from pathlib import Path
import numpy as np
import torch
from hypermil.config import load_config,resolve,symbol,ROOT
from hypermil.data import InputStore
from hypermil.metrics import harrell_c_index

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def filename(config,cohort,fold):
    return config['checkpoint']['pattern'].format(cohort=cohort,cohort_lower=cohort.lower(),fold=fold)

@torch.inference_mode()
def evaluate(config_path,cohort,fold,device,checkpoint=None,output=None):
    config=load_config(config_path);method=config['method']
    checkpoint=Path(checkpoint or resolve(config['checkpoint']['directory'])/filename(config,cohort,fold))
    digest=sha256(checkpoint)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert payload['method']==method and payload['cohort']==cohort and payload['fold']==fold
    split_hash=sha256(resolve(config['data']['patient_folds']))
    assert split_hash==payload['fold_manifest_sha256'], 'Split manifest mismatch'
    store=InputStore(config)
    patients=store.patients(cohort,fold,config['evaluation']['split'])
    adapter=symbol(config['interface'])(config,store,cohort,fold,torch.device(device))
    adapter.load(payload);adapter.eval()
    values=np.asarray(adapter.predict(patients),dtype=np.float64)
    assert values.shape==(len(patients),) and np.isfinite(values).all()
    score=harrell_c_index([p.time for p in patients],[p.event for p in patients],values)
    out=Path(output or ROOT/'results/eval'/method/cohort/f'fold{fold}')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['patient_id','time','event','risk'])
        writer.writeheader()
        writer.writerows({'patient_id':p.patient_id,'time':p.time,'event':p.event,'risk':float(r)} for p,r in zip(patients,values))
    result={'method':method,'cohort':cohort,'fold':fold,'patients':len(patients),
        'cindex':score,'checkpoint':str(checkpoint),'checkpoint_sha256':digest,
        'split_sha256':split_hash,'strict_load':True,'fresh_forward':True,
        'engine_sha256':sha256(Path(__file__)),'interface_sha256':sha256(ROOT/'hypermil/interfaces.py'),
        'config_sha256':sha256(resolve(config_path)),'max_patches':config['data']['max_patches']}
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return result
