import importlib
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]

def resolve(path):
    path=Path(path)
    return path if path.is_absolute() else ROOT/path

def load_config(path):
    cfg=yaml.safe_load(resolve(path).read_text())
    assert isinstance(cfg,dict) and all(k in cfg for k in ('method','data','model','interface','evaluation'))
    return cfg

def symbol(name):
    module,attr=name.split(':');return getattr(importlib.import_module(module),attr)

def build_model(config,device):
    from hypermil.adapters import CoreWrapper
    spec=config['model'];args=dict(spec.get('args',{}))
    if spec.get('device_argument'):args['device']=device
    model=symbol(spec['factory'])(**args)
    if spec.get('wrap_core'):model=CoreWrapper(model)
    return model.to(device)
