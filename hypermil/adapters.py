"""Method-specific model loading and risk prediction; no evaluation metrics here."""
from __future__ import annotations
import json, types
from pathlib import Path
import numpy as np
import torch
from torch import nn
from hypermil.models.hypermil import make_hypermil, prepare, GatedMILAttention

def strict(model,state,device):
    model.load_state_dict(state,strict=True)
    return model.to(device).eval()

def ranks(x):
    order=np.argsort(x,kind='mergesort');out=np.empty(len(x),dtype=np.float64);i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]: j+=1
        out[order[i:j]]=.5*(i+j-1);i=j
    return out/max(1,len(x)-1)

def modify_incidence(raw,variant,family,cohort,sid):
    if variant=='A3' and family=='mgas':
        rows,cols,values=raw.rows.numpy(),raw.cols.numpy(),raw.values.numpy()
        sizes=np.bincount(cols,minlength=raw.shape[1]);order=np.lexsort((cols,sizes[cols],rows))
        first=np.r_[True,rows[order][1:]!=rows[order][:-1]];leaves=np.unique(cols[order[first]])
        keep=np.isin(cols,leaves);mapping=np.full(raw.shape[1],-1,dtype=np.int64);mapping[leaves]=np.arange(len(leaves))
        return type(raw)(rows[keep].copy(),mapping[cols[keep]].copy(),values[keep].copy(),(raw.shape[0],len(leaves)))
    if variant=='A4' and family=='pasg':
        rows,cols,values=raw.rows.numpy(),raw.cols.numpy(),raw.values.numpy()
        order=np.lexsort((cols,-values,rows));chosen=order[np.r_[True,rows[order][1:]!=rows[order][:-1]]]
        used=np.unique(cols[chosen]);mapping=np.full(raw.shape[1],-1,dtype=np.int64);mapping[used]=np.arange(len(used))
        return type(raw)(rows[chosen].copy(),mapping[cols[chosen]].copy(),np.ones(len(chosen),np.float32),(raw.shape[0],len(used)))
    return raw

def single_encode(self,x,mgas,pasg):
    if self._variant=='A1':
        z=self.mgas_norm(self.mgas_branch(x,mgas));emb,att=self.attention_s(self.pre_attention_norm(z))
    else:
        z=self.pasg_norm(self.pasg_branch(x,pasg));emb,att=self.attention_f(self.pre_attention_norm(z))
    return types.SimpleNamespace(embedding=emb,attention=att)

def fixed_encode(self,x,mgas,pasg):
    a=self.mgas_norm(self.mgas_branch(x,mgas));b=self.pasg_norm(self.pasg_branch(x,pasg))
    a,att=self.attention_s(self.pre_attention_norm(a));b,_=self.attention_f(self.pre_attention_norm(b))
    return types.SimpleNamespace(embedding=.5*a+.5*b,attention=att)

def group_mlp(branch,x):
    for layer in branch.layers: x=branch.hidden_dropout(branch.activation(layer.linear(x)))
    return x

def group_encode(self,x,mgas,pasg):
    a=self.mgas_norm(group_mlp(self.mgas_branch,mgas));b=self.pasg_norm(group_mlp(self.pasg_branch,pasg))
    a,att=self.attention_s(self.pre_attention_norm(a));b,_=self.attention_f(self.pre_attention_norm(b))
    g=torch.sigmoid(self.wsi_fusion(torch.cat((a,b),dim=-1)))
    return types.SimpleNamespace(embedding=g*a+(1-g)*b)

def weighted_groups(x,raw):
    rows,cols,v=raw.rows.cpu(),raw.cols.cpu(),raw.values.cpu()
    degree=torch.zeros(raw.shape[1],dtype=torch.float32).index_add_(0,cols,v)
    out=torch.zeros(raw.shape[1],x.shape[1],dtype=torch.float32)
    for i in range(0,len(v),16384):
        j=i+16384;out.index_add_(0,cols[i:j],v[i:j,None]*x[rows[i:j]])
    return out/degree[:,None]

class CoreWrapper(nn.Module):
    def __init__(self,core): super().__init__();self.core=core

def ablation_model(variant,device):
    core=make_hypermil(device)
    if variant in ('A1','A2'):
        names=('pasg_branch','pasg_norm','attention_f','wsi_fusion') if variant=='A1' else ('mgas_branch','mgas_norm','attention_s','wsi_fusion')
        for name in names: delattr(core,name)
        core._variant=variant;core.encode_slide=types.MethodType(single_encode,core)
    elif variant=='A5':
        del core.wsi_fusion;core.encode_slide=types.MethodType(fixed_encode,core)
    elif variant=='A6': core.encode_slide=types.MethodType(group_encode,core)
    return core

class PatientAttention(nn.Module):
    def __init__(self,device):
        super().__init__();self.base=make_hypermil(device);self.patient_attention=GatedMILAttention(256).to(device)
