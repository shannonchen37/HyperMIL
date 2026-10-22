"""Model interfaces selected by YAML. Splits, feature loading and metrics are shared."""
from pathlib import Path
import numpy as np
import torch
from hypermil.config import build_model,symbol
from hypermil.adapters import strict,ranks,prepare,modify_incidence,weighted_groups

class ModelInterface:
    def __init__(self,config,store,cohort,fold,device):
        self.config,self.store,self.cohort,self.fold,self.device=config,store,cohort,fold,device
        self.params=config.get('parameters',{});self.cache={}
        self.model=build_model(config,device)

    def load(self,payload):strict(self.model,payload['model_state'],self.device)
    def eval(self):self.model.eval();return self
    def tensor_risks(self,patients):return torch.stack([self.patient_risk(p,i) for i,p in enumerate(patients)])
    def predict(self,patients):return self.tensor_risks(patients).detach().cpu().double().numpy()
    def training_stages(self):return [{'name':'model','model':self.model}]
    def activate_training_stage(self,stage):pass
    def training_payload(self):
        from hypermil.engine import sha256
        from hypermil.config import resolve
        return {'format':'hypermil_release_v1','method':self.config['method'],'cohort':self.cohort,
            'fold':self.fold,'model_state':self.model.state_dict(),
            'fold_manifest_sha256':sha256(resolve(self.config['data']['patient_folds']))}

class MILInterface(ModelInterface):
    def patient_risk(self,p,index):
        z=torch.stack([self.model.encode_slide(x) for x in self.store.bags(p,self.device)]).mean(0)
        return self.model.risk_head(z).reshape(())

class DSMILInterface(ModelInterface):
    def patient_risk(self,p,index):
        encoded=[self.model.encode_slide(x) for x in self.store.bags(p,self.device)]
        zi=torch.stack([x[0] for x in encoded]).mean(0);zb=torch.stack([x[1] for x in encoded]).mean(0)
        ri=self.model.instance_head(zi).reshape(());rb=self.model.bag_head(zb.view(1,1,512)).reshape(())
        return torch.stack((ri,rb)).mean()

class ACMILInterface(ModelInterface):
    def patient_risk(self,p,index):return self.model.forward_patient_bags(self.store.bags(p,self.device)).main_risk

class DTFDInterface(ModelInterface):
    def patient_risk(self,p,index):
        # Reproduce the saved protocol's fixed validation partition schedule.
        if not self.model.training:
            seed=int(self.params['partition_seed_base'])+index
            torch.manual_seed(seed);torch.cuda.manual_seed(seed)
        return self.model.tier2_patient(self.store.bags(p,self.device)).risk

class AttriMILInterface(ModelInterface):
    def patient_risk(self,p,index):
        logits=[]
        for x in self.store.bags(p,self.device):
            with torch.device(self.device):out,*_=self.model.core(x)
            logits.append(out.squeeze(0))
        z=torch.stack(logits).mean(0);return z[1]-z[0]

class PatchGCNInterface(ModelInterface):
    def patient_risk(self,p,index):
        zs=[]
        for s in p.slides:
            graph=torch.load(Path(self.store.paths[self.params['graph_directory_key']])/f'TCGA-{self.cohort}'/f'{s.sample_id}.pt',map_location='cpu',weights_only=False)
            x,_=self.store.load(s,self.device)
            assert torch.equal(graph.x,x.cpu()),f'PatchGCN feature mismatch {s.sample_id}'
            zs.append(self.model.encode_slide(x,edge_index=graph.edge_index.to(self.device)))
        return self.model.risk_head(torch.stack(zs).mean(0)).reshape(())

class DualGCNInterface(ModelInterface):
    def patient_risk(self,p,index):
        zs=[]
        for s in p.slides:
            base=Path(self.store.paths[self.params['graph_directory_key']])/f'TCGA-{self.cohort}'
            kc,kf=self.params['spatial_k'],self.params['feature_k']
            with np.load(base/'spatial'/f'{s.sample_id}_knn_spatial_k{kc}.npz') as z:ec=torch.from_numpy(z['edge_index'].copy()).long().to(self.device)
            with np.load(base/'feature'/f'{s.sample_id}_knn_feature_k{kf}.npz') as z:ef=torch.from_numpy(z['edge_index'].copy()).long().to(self.device)
            x,_=self.store.load(s,self.device);zc=self.model.branch_coord(x,ec);zf=self.model.branch_feat(x,ef)
            zs.append(torch.cat([zf,zc],dim=0))
        return self.model.risk_head(torch.stack(zs).mean(0)[None]).reshape(())

class KMeansInterface(ModelInterface):
    def patient_risk(self,p,index):
        zs=[]
        for s in p.slides:
            # Cache contains only model-independent aggregation of the same full features.
            x,_=self.store.arrays(s)
            assert len(x)==s.n_patches
            with np.load(Path(self.store.paths['kmeans'])/self.params['cache_subdir']/'cache'/self.cohort/f'{s.sample_id}.npz') as z:
                assert int(z['counts'].sum())==s.n_patches
                zs.append(np.array(z['slide_embedding'],dtype=np.float32))
        return self.model(torch.from_numpy(np.stack(zs).mean(0)[None]).to(self.device)).reshape(())

class HypergraphInterface(ModelInterface):
    def slide_inputs(self,s):
        variant=self.params.get('variant','A0')
        x,_=self.store.load(s,'cpu' if self.params.get('aggregation')=='incidence_group_mean' else self.device)
        raw=[modify_incidence(self.store.incidence(self.cohort,f,s.sample_id),variant,f,self.cohort,s.sample_id) for f in ('mgas','pasg')]
        if self.params.get('aggregation')=='incidence_group_mean':return None,*[weighted_groups(x,r).to(self.device) for r in raw]
        return x,*[prepare(r,self.device) for r in raw]

    def core_risk(self,model,p,attention=False):
        core=model.base if attention else model
        zs=torch.stack([core.encode_slide(*self.slide_inputs(s)).embedding for s in p.slides])
        z=model.patient_attention(zs)[0] if attention else zs.mean(0)
        return core.risk_head(z).reshape(())

    def patient_risk(self,p,index):
        core=self.model.core if self.config['model'].get('wrap_core') else self.model
        return self.core_risk(core,p)

class PipelineInterface(HypergraphInterface):
    def load(self,payload):self.payload=payload
    def training_payload(self):
        payload=super().training_payload();state=payload.pop('model_state')
        payload.update({'members':[{'architecture':'canonical','weight':1.,'model_state':state}],
            'combination_rule':'raw_single_model_risk'})
        return payload
    def predict(self,patients):
        vectors=[]
        for member in self.payload['members']:
            attention=member['architecture']=='patient_attention'
            factory=self.params['attention_factory' if attention else 'canonical_factory']
            model=symbol(factory)(self.device)
            strict(model,member['model_state'],self.device)
            vectors.append(torch.stack([self.core_risk(model,p,attention) for p in patients]).cpu().double().numpy())
            del model;torch.cuda.empty_cache()
        rule=self.payload['combination_rule']
        if rule=='raw_single_model_risk':return vectors[0]
        if rule!='within_fold_percentile_rank_weighted_sum':raise ValueError(rule)
        return np.stack([ranks(x) for x in vectors],axis=1)@np.asarray([m['weight'] for m in self.payload['members']],dtype=np.float64)

class HyperAdaCInterface(ModelInterface):
    def embedding(self,p):
        from hypermil.models.hyper_adac import indexed_adapter,reduce_graph
        graphs=[]
        for s in p.slides:
            x,c=self.store.arrays(s)
            labels,_=indexed_adapter(np.asarray(x,dtype=np.float32),np.asarray(c,dtype=np.float64),
                ld=self.params['lambda_d'],lf=self.params['lambda_f'],th=self.params['cluster_threshold'])
            rx,_,_,edge=reduce_graph(x,c,labels)
            graphs.append((torch.from_numpy(rx).to(self.device),edge.to(self.device)))
        return self.model.patient_embedding(graphs)
    def tensor_risks(self,patients):
        # BatchNorm in the original survival head is applied to patient embeddings.
        return self.model.patient_risks(torch.stack([self.embedding(p) for p in patients]))
    def patient_risk(self,p,index):return self.model.patient_risks(self.embedding(p)[None]).reshape(())

class IIHGCInterface(ModelInterface):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.intra=symbol(self.params['stage1_factory'])().to(self.device)
        self.stage='inter'
    def load(self,payload):
        super().load(payload);strict(self.intra,payload['stage1_state'],self.device)
    def eval(self):super().eval();self.intra.eval();return self
    def intra_patient(self,p):
        slides=[]
        for s in p.slides:
            x,c=self.store.load(s,self.device);slides.append((x[None],c.to(torch.float32)[None]))
        return self.intra.patient(slides,train=self.intra.training)
    def patient_risk(self,p,index):
        if self.stage=='intra':return self.intra_patient(p)[0]
        if 'train_context' not in self.cache:
            train=self.store.patients(self.cohort,self.fold,'train')
            pairs=[self.intra_patient(p) for p in train]
            self.cache['train_context']=(torch.stack([x[1] for x in pairs]).detach(),torch.stack([x[0] for x in pairs])[:,None].detach())
        zt,rt=self.cache['train_context'];r,z=self.intra_patient(p)
        return self.model(z[None],r.reshape(1,1),zt,rt)[1].reshape(())
    def training_stages(self):
        return [{'name':'intra','model':self.intra,'epochs':self.params['stage1_epochs']},{'name':'inter','model':self.model}]
    def activate_training_stage(self,stage):
        self.stage=stage['name'];self.cache.clear()
        if self.stage=='inter':
            self.intra.eval()
            for p in self.intra.parameters():p.requires_grad_(False)
    def training_payload(self):
        payload=super().training_payload();payload['stage1_state']=self.intra.state_dict();return payload
