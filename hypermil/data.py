from __future__ import annotations
import csv,json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import yaml

@dataclass(frozen=True)
class Slide:
    sample_id: str
    patient_id: str
    feature_path: Path
    coordinate_path: Path
    n_patches: int

@dataclass(frozen=True)
class Patient:
    patient_id: str
    slides: tuple[Slide, ...]
    time: float
    event: int

class InputStore:
    """One configured feature/coordinate loader for all model interfaces."""

    def __init__(self, config):
        from hypermil.config import resolve
        self.config = config
        self.data = config['data']
        self.paths = yaml.safe_load(resolve(config['paths_file']).read_text())
        self.indices = None
        if self.data['max_patches'] is not None:
            with resolve(self.data['patch_selection']).open(newline='') as f:
                self.indices = {r['sample_id']: np.fromstring(r['indices'], sep=' ', dtype=np.int64) for r in csv.DictReader(f)}
        self._raw_cache = {}

    def slides(self, cohort):
        from hypermil.config import resolve
        out = []
        with resolve(self.data['feature_manifest']).open(newline='') as f:
            for r in csv.DictReader(f):
                if r['cohort'] != cohort:
                    continue
                ft, co = (Path(r['feature_path']), Path(r['coordinate_path']))
                if self.paths.get('wsi_data'):
                    base = Path(self.paths['wsi_data']) / f'TCGA-{cohort}'
                    ft = base / 'patch_ft' / ft.name
                    co = base / 'patch_coor' / co.name
                out.append(Slide(r['sample_id'], r['patient_id'], ft, co, int(r['n_patches'])))
        return tuple(sorted(out, key=lambda s: s.sample_id))

    def patients(self, cohort, fold, split='outer_val'):
        from hypermil.config import resolve
        by = {}
        for slide in self.slides(cohort):
            by.setdefault(slide.patient_id, []).append(slide)
        with resolve(self.data['patient_folds']).open(newline='') as f:
            rows = [r for r in csv.DictReader(f) if r['cohort'] == cohort and int(r['fold']) == fold]
        assert len(rows) == len(by) == len({r['patient_id'] for r in rows})
        selected = [r for r in rows if (r['split'] != 'outer_val' if split == 'train' else r['split'] == split)]
        return tuple(sorted((Patient(r['patient_id'], tuple(by[r['patient_id']]), float(r['time']), int(r['event'])) for r in selected), key=lambda p: p.patient_id))

    def arrays(self, slide):
        x = np.load(slide.feature_path, mmap_mode='r')
        c = np.load(slide.coordinate_path, mmap_mode='r')
        assert x.shape == (slide.n_patches, self.data['feature_dim']) and str(x.dtype) == self.data['feature_dtype']
        assert len(c) == slide.n_patches
        limit = self.data['max_patches']
        if limit is not None:
            idx = self.indices[slide.sample_id]
            assert len(idx) == min(int(limit), slide.n_patches) and len(set(idx)) == len(idx)
            assert (idx >= 0).all() and (idx < slide.n_patches).all()
            x, c = (x[idx], c[idx])
        return (x, c)

    def load(self, slide, device):
        x, c = self.arrays(slide)
        return (torch.from_numpy(np.asarray(x)).to(device=device, dtype=torch.float32), torch.from_numpy(np.asarray(c)).to(device))

    def bags(self, patient, device):
        return [self.load(s, device)[0] for s in patient.slides]

    def graph_path(self, cohort, family, sid):
        name = 'MGAS_V2' if family == 'mgas' else 'PASG_V2_FINAL'
        return Path(self.paths['hypergraphs']) / f'{cohort}_{name}' / f'{sid}.npz'

    def incidence(self, cohort, family, sid):
        from hypermil.models.hypermil import SparseIncidence
        path = self.graph_path(cohort, family, sid)
        if family == 'mgas':
            return SparseIncidence.load(path)
        with np.load(path, allow_pickle=False) as z:
            return SparseIncidence(z['incidence_row'].copy(), z['incidence_col'].copy(), z['incidence_weight'].copy(), tuple((int(x) for x in z['incidence_shape'])))

