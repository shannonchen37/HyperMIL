"""One full-patient-risk-set training loop; model stages belong to interfaces."""
import json,random
from pathlib import Path
import numpy as np
import torch
from hypermil.config import load_config,symbol
from hypermil.data import InputStore
from hypermil.cox import cox_efron_loss
from hypermil.metrics import harrell_c_index

def train(config_path,cohort,fold,device,output):
    cfg=load_config(config_path);tcfg=cfg['training'];device=torch.device(device)
    seed=int(tcfg['seed']);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    store=InputStore(cfg);interface=symbol(cfg['interface'])(cfg,store,cohort,fold,device)
    train_set=store.patients(cohort,fold,'inner_train')
    val_set=store.patients(cohort,fold,tcfg['selection_split'])
    assert train_set and val_set
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'best.pt').exists():raise FileExistsError('Use a new training output directory')
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    t=torch.tensor([p.time for p in train_set],device=device)
    e=torch.tensor([p.event for p in train_set],device=device)
    stages=interface.training_stages()
    for stage in stages:
        model=stage['model'];interface.activate_training_stage(stage)
        opt=getattr(torch.optim,tcfg['optimizer'])(model.parameters(),lr=float(tcfg['lr']),weight_decay=float(tcfg['weight_decay']))
        best=-float('inf')
        for epoch in range(1,int(stage.get('epochs',tcfg['epochs']))+1):
            model.train();opt.zero_grad(set_to_none=True)
            # Complete training risk set; interfaces may retain their published auxiliary objectives.
            risks=interface.tensor_risks(train_set)
            loss=cox_efron_loss(risks,t,e)
            loss.backward();opt.step();interface.eval()
            with torch.no_grad():values=interface.tensor_risks(val_set).cpu().double().numpy()
            c=harrell_c_index([p.time for p in val_set],[p.event for p in val_set],values)
            record={'stage':stage['name'],'epoch':epoch,'train_loss':float(loss.detach()),'validation_cindex':c}
            with (out/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if c>best:
                best=c;torch.save({'model_state':model.state_dict(),**record},out/f"{stage['name']}_best.pt")
            print(json.dumps(record),flush=True)
        payload=torch.load(out/f"{stage['name']}_best.pt",map_location=device,weights_only=False)
        model.load_state_dict(payload['model_state'],strict=True);model.eval()
    torch.save(interface.training_payload(),out/'best.pt')
