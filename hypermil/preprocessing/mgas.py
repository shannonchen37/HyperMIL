from __future__ import annotations

import math
import time
from collections import deque

import networkx as nx
import numpy as np
import torch
from scipy.sparse import diags, eye
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree

from hypermil.models.hypermil import SparseIncidence


def build_base_graph(coordinates, features, tau_s=2.9, tau_c=0.75, device="cpu", patch_size=256.0):
    started=time.perf_counter()
    coords=np.asarray(coordinates,dtype=np.float64)/float(patch_size)
    feats=np.asarray(features)
    pairs=cKDTree(coords).query_pairs(float(tau_s),output_type="ndarray")
    candidates=len(pairs)
    if len(pairs):
        delta=coords[pairs[:,0]]-coords[pairs[:,1]]
        pairs=pairs[np.einsum("ij,ij->i",delta,delta)>0]
        kept=[]; dev=torch.device(device)
        for start in range(0,len(pairs),100_000):
            q=pairs[start:start+100_000]
            a=torch.as_tensor(np.asarray(feats[q[:,0]],dtype=np.float32),device=dev)
            b=torch.as_tensor(np.asarray(feats[q[:,1]],dtype=np.float32),device=dev)
            sim=torch.nn.functional.cosine_similarity(a,b,dim=1)
            kept.append(q[(sim>=tau_c).detach().cpu().numpy()])
        pairs=np.concatenate(kept) if kept else np.empty((0,2),dtype=np.int64)
    graph=nx.Graph();graph.add_nodes_from(range(len(coords)));graph.add_edges_from(map(tuple,pairs.tolist()))
    return graph,{"candidate_edges":int(candidates),"base_edges":int(len(pairs)),"base_graph_seconds":time.perf_counter()-started}


def _bfs_partition(graph,centers):
    groups={c:set() for c in centers};queues={c:deque([c]) for c in centers};visited={c:c for c in centers}
    active=True
    while active:
        active=False
        for center in centers:
            if queues[center]:
                active=True;node=queues[center].popleft();groups[center].add(node)
                for neighbor in sorted(graph.neighbors(node)):
                    if neighbor not in visited:visited[neighbor]=center;queues[center].append(neighbor)
    for node in graph.nodes:
        if node not in visited:groups[centers[0]].add(node)
    return list(groups.values())


def _sparse_spectral(region):
    nodes=sorted(region.nodes());n=len(nodes)
    if n<2:return None
    adjacency=nx.to_scipy_sparse_array(region,nodelist=nodes,dtype=float,format="csr")
    degree=np.asarray(adjacency.sum(1)).ravel()
    if np.any(degree<=0):return None
    inv=diags(1/np.sqrt(degree));lap=eye(n,format="csr")-inv@adjacency@inv
    try:
        _,vectors=eigsh(lap,k=2,which="SM",v0=np.ones(n),tol=1e-8,maxiter=max(1000,n*10))
    except Exception:
        return None
    fiedler=vectors[:,1];threshold=float(np.median(fiedler))
    left={nodes[i] for i in range(n) if fiedler[i]<=threshold};right=set(nodes)-left
    return (left,right) if left and right else None


def _ncut(graph,left,right):
    cut=sum(1 for u,v in graph.edges() if (u in left)!=(v in left))
    vol_left=sum(dict(graph.degree(left)).values());vol_right=sum(dict(graph.degree(right)).values())
    return float("inf") if min(vol_left,vol_right)==0 else cut/vol_left+cut/vol_right


def segment_mgas(graph,max_depth=2,min_region_size=2,ncut_threshold=0.8):
    started=time.perf_counter();edges=[];depths=[];parents=[];accepted=[];seen=set()
    def add(nodes,depth,parent):
        key=tuple(sorted(nodes))
        if not key or key in seen:raise RuntimeError("empty or duplicate structural hyperedge")
        seen.add(key);edges.append(key);depths.append(depth);parents.append(parent);return len(edges)-1
    def recurse(region,depth,parent):
        nodes=set(region.nodes());edge_id=add(nodes,depth,parent)
        if len(nodes)<2*min_region_size or depth>=max_depth:return edge_id
        partition=_sparse_spectral(region)
        if not partition:return edge_id
        left,right=partition
        if min(len(left),len(right))<min_region_size:return edge_id
        score=_ncut(region,left,right)
        if not score<=ncut_threshold:return edge_id
        left_id=recurse(region.subgraph(left).copy(),depth+1,edge_id)
        right_id=recurse(region.subgraph(right).copy(),depth+1,edge_id)
        accepted.append({"parent":edge_id,"left":left_id,"right":right_id,"ncut":float(score)})
        return edge_id
    for component in nx.connected_components(graph):
        cg=graph.subgraph(component).copy()
        if len(cg)==1:add(component,0,-1);continue
        count=max(1,math.isqrt(len(cg)));centers=sorted(component,key=lambda x:(-cg.degree(x),x))[:count]
        for group in _bfs_partition(cg,centers):recurse(cg.subgraph(group).copy(),0,-1)
    incidence=SparseIncidence.from_hyperedges(edges,graph.number_of_nodes())
    counts=torch.zeros(graph.number_of_nodes(),dtype=torch.long)
    counts.scatter_add_(0,incidence.rows,torch.ones_like(incidence.rows))
    for split in accepted:
        p=set(edges[split["parent"]]);left=set(edges[split["left"]]);right=set(edges[split["right"]])
        if left&right or left|right!=p:raise RuntimeError("child partition does not reconstruct parent")
    stats=incidence.statistics()|{"n_nodes":graph.number_of_nodes(),"n_hyperedges":len(edges),"incidence_nnz":int(incidence.values.numel()),
      "membership_median":float(counts.float().median()) if len(counts) else 0.0,"pct_membership_gt1":float((counts>1).float().mean()*100) if len(counts) else 0.0,
      "depth_distribution":{str(d):depths.count(d) for d in sorted(set(depths))},"accepted_splits":len(accepted),
      "max_depth":max(depths,default=0),"build_seconds":time.perf_counter()-started}
    hierarchy={"depths":depths,"parents":parents,"accepted_splits":accepted}
    return incidence,hierarchy,stats
