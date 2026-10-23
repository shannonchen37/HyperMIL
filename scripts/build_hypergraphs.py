#!/usr/bin/env python3
"""Build label-independent MGAS/PASG caches using the shared feature loader."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from hypermil.config import load_config, resolve
from hypermil.data import InputStore
from hypermil.preprocessing.mgas import build_base_graph, segment_mgas
from hypermil.preprocessing.pasg import build_pasg, save_pasg, configure_determinism


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/preprocessing/hypergraphs.yaml')
    parser.add_argument('--cohort', required=True, type=str.upper, choices=['STAD', 'THCA', 'CHOL', 'LIHC'])
    parser.add_argument('--family', default='BOTH', type=str.upper, choices=['BOTH', 'MGAS', 'PASG'])
    parser.add_argument('--slide', help='Build only this slide ID from the feature manifest')
    parser.add_argument('--device', help='Override runtime.device from YAML')
    parser.add_argument('--output-root', help='Optional relative output directory')
    parser.add_argument('--overwrite', action='store_true', help='Rebuild existing graph files')
    args = parser.parse_args(argv)
    config_path = resolve(args.config)
    cfg = yaml.safe_load(config_path.read_text())
    store = InputStore(load_config(cfg['input_config']))
    if store.data['max_patches'] is not None:
        raise ValueError('HyperMIL graph construction requires complete WSI features')
    slides = store.slides(args.cohort)
    if args.slide:
        slides = tuple(s for s in slides if s.sample_id == args.slide)
    if not slides:
        raise ValueError('No matching slides in the feature manifest')
    output = resolve(args.output_root or cfg['output_root'])
    device = args.device or cfg['runtime']['device']
    configure_determinism()
    for slide in slides:
        x = coordinates = None
        for family in (['MGAS', 'PASG'] if args.family == 'BOTH' else [args.family]):
            suffix = 'MGAS_V2' if family == 'MGAS' else 'PASG_V2_FINAL'
            path = output / f'{args.cohort}_{suffix}' / f'{slide.sample_id}.npz'
            if path.exists() and not args.overwrite:
                print(json.dumps(dict(slide=slide.sample_id, family=family, status='existing')), flush=True)
                continue
            if x is None:
                x, coordinates = store.arrays(slide)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.partial.npz')
            if family == 'MGAS':
                settings = cfg['mgas']
                graph, base = build_base_graph(coordinates, x, tau_s=settings['spatial_radius'],
                    tau_c=settings['cosine_threshold'], patch_size=settings['patch_size'], device=device)
                incidence, hierarchy, stats = segment_mgas(graph, max_depth=settings['max_depth'],
                    min_region_size=settings['min_region_size'], ncut_threshold=settings['ncut_threshold'])
                incidence.save(temporary)
                report = dict(**base, **stats, hierarchy=hierarchy)
            else:
                result = build_pasg(x, slide.sample_id, cfg['pasg'], device=device)
                save_pasg(result, temporary)
                report = result.metadata
            temporary.replace(path)
            report.update(slide_id=slide.sample_id, family=family,
                          config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest())
            path.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(dict(slide=slide.sample_id, family=family, status='built',
                                  output=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path))), flush=True)


if __name__ == '__main__':
    main()
