#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build PASG hypergraphs with GPU-accelerated Bayesian GMM clustering."""

import numpy as np
import os
import argparse
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from tqdm import tqdm
import time
import pickle
from collections import defaultdict
from sklearn.cluster import KMeans

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def resolve_device(device_name):
    """Parse the device name from the command line."""
    if isinstance(device_name, torch.device):
        return device_name

    if device_name in (None, 'auto'):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    device_name = str(device_name)
    if device_name.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but no GPU is available.")

    return torch.device(device_name)


class BayesianGMMGPU:
    """GPU-accelerated Bayesian Gaussian Mixture Model with full covariance."""

    def __init__(self, n_components=10,
                 weight_concentration_prior=0.1, max_iter=100,
                 tol=1e-3, random_state=42, device=None):
        """Initialize the variational Bayesian GMM."""
        self.n_components = n_components
        self.covariance_type = 'full'
        self.weight_concentration_prior = weight_concentration_prior
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
        self.converged_ = False
        self.n_iter_ = 0

    def _initialize_parameters(self, X):
        """Initialize mixture parameters with K-means."""
        N, D = X.shape
        X_cpu = X.cpu().numpy()
        kmeans = KMeans(n_clusters=self.n_components, random_state=self.random_state, n_init=1)
        labels = kmeans.fit_predict(X_cpu)

        self.means_ = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=self.device)
        self.covariances_ = []
        for k in range(self.n_components):
            mask = labels == k
            if mask.sum() > 1:
                X_k = X_cpu[mask]
                cov = np.cov(X_k.T) + 1e-6 * np.eye(D)
            else:
                cov = np.eye(D)
            self.covariances_.append(torch.tensor(cov, dtype=torch.float32, device=self.device))

        self.weights_ = torch.ones(self.n_components, dtype=torch.float32, device=self.device) / self.n_components

    def _e_step(self, X):
        """Expectation step."""
        N, D = X.shape
        K = self.n_components
        log_prob = torch.zeros(N, K, device=self.device)

        for k in range(K):
            diff = X - self.means_[k]
            cov = self.covariances_[k]
            cov_reg = cov + 1e-6 * torch.eye(D, device=self.device)
            try:
                L = torch.linalg.cholesky(cov_reg)
            except:
                cov_reg = cov_reg + 1e-3 * torch.eye(D, device=self.device)
                L = torch.linalg.cholesky(cov_reg)
            log_det = 2 * torch.sum(torch.log(torch.diagonal(L)))
            solve = torch.linalg.solve_triangular(L, diff.T, upper=False)
            maha = torch.sum(solve ** 2, dim=0)

            log_prob[:, k] = -0.5 * (D * np.log(2 * np.pi) + log_det + maha)
        log_prob += torch.log(self.weights_ + 1e-10)
        log_prob_norm = torch.logsumexp(log_prob, dim=1)
        log_resp = log_prob - log_prob_norm.unsqueeze(1)
        resp = torch.exp(log_resp)

        return resp, log_prob_norm.sum()

    def _m_step(self, X, resp):
        """Maximization step."""
        N, D = X.shape
        K = self.n_components
        nk = resp.sum(dim=0) + 1e-10
        weight_concentration = self.weight_concentration_prior
        self.weights_ = (nk + weight_concentration) / (N + K * weight_concentration)
        self.means_ = torch.matmul(resp.T, X) / nk.unsqueeze(1)

        for k in range(K):
            diff = X - self.means_[k]
            weighted_diff = diff * resp[:, k].unsqueeze(1).sqrt()
            cov = torch.matmul(weighted_diff.T, weighted_diff) / nk[k]
            cov = cov + 1e-6 * torch.eye(D, device=self.device)
            self.covariances_[k] = cov

    def fit(self, X):
        """Fit the model."""
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)

        N, D = X.shape
        self._initialize_parameters(X)
        prev_log_likelihood = -np.inf

        for i in range(self.max_iter):
            resp, log_likelihood = self._e_step(X)
            self._m_step(X, resp)
            improvement = log_likelihood - prev_log_likelihood

            if abs(improvement) < self.tol and i > 10:
                self.converged_ = True
                self.n_iter_ = i + 1
                break

            prev_log_likelihood = log_likelihood

        if not self.converged_:
            self.n_iter_ = self.max_iter

        return self

    def predict(self, X):
        """Predict cluster labels."""
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        resp, _ = self._e_step(X)
        labels = torch.argmax(resp, dim=1)
        return labels.cpu().numpy()

    def get_centroids(self):
        """Return cluster centroids."""
        return self.means_.cpu().numpy()

    def get_weights(self):
        """Return cluster weights."""
        return self.weights_.cpu().numpy()


def discover_feature_files(feature_dir):
    """Discover feature files directly from `patch_ft`."""
    print(f"Scanning feature directory: {feature_dir}")

    if not os.path.isdir(feature_dir):
        raise FileNotFoundError(f"Feature directory not found: {feature_dir}")

    file_paths = []
    for name in sorted(os.listdir(feature_dir)):
        if name.endswith("_fts.npy"):
            file_paths.append(os.path.join(feature_dir, name))

    print(f"  Found {len(file_paths)} feature files")
    return file_paths


def build_hyperedges(assignments, unique_clusters):
    """Build hyperedges from cluster assignments."""
    hyperedge_dict = defaultdict(list)

    for patch_idx, cluster_id in enumerate(assignments):
        hyperedge_dict[cluster_id].append(patch_idx)

    hyperedges = [hyperedge_dict[cid] for cid in unique_clusters]

    return hyperedges, hyperedge_dict


def process_single_wsi_gpu(feat_file, output_dir, n_components=10,
                           concentration=0.1, max_iter=100, skip_existing=True,
                           device=None):
    """Process a single WSI feature file and save PASG hypergraph outputs."""
    start_time = time.time()
    basename = os.path.basename(feat_file)
    wsi_name = os.path.splitext(basename)[0]

    wsi_output_dir = os.path.join(output_dir, wsi_name)
    if skip_existing and os.path.exists(wsi_output_dir):
        required_files = [
            'cluster_assignments.npy',
            'cluster_centroids.npy',
            'hyperedges.pkl',
            'report.txt'
        ]

        all_exist = all(os.path.exists(os.path.join(wsi_output_dir, f))
                       for f in required_files)

        if all_exist:
            return {
                'wsi_name': wsi_name,
                'status': 'skipped',
                'message': 'Already processed',
                'time': 0
            }

    try:
        features = np.load(feat_file)
        n_patches = features.shape[0]
        feat_dim = features.shape[1]

        model_gpu = BayesianGMMGPU(
            n_components=n_components,
            weight_concentration_prior=concentration,
            max_iter=max_iter,
            random_state=42,
            device=device
        )

        model_gpu.fit(features)
        assignments = model_gpu.predict(features)
        centroids = model_gpu.get_centroids()

        unique_clusters = np.unique(assignments)
        hyperedges, hyperedge_dict = build_hyperedges(assignments, unique_clusters)

        os.makedirs(wsi_output_dir, exist_ok=True)
        np.save(os.path.join(wsi_output_dir, 'cluster_assignments.npy'), assignments)
        np.save(os.path.join(wsi_output_dir, 'cluster_centroids.npy'), centroids)
        with open(os.path.join(wsi_output_dir, 'hyperedges.pkl'), 'wb') as f:
            pickle.dump(hyperedges, f)

        elapsed_time = time.time() - start_time
        with open(os.path.join(wsi_output_dir, 'report.txt'), 'w', encoding='utf-8') as f:
            f.write(f"WSI: {wsi_name}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Patches: {n_patches}\n")
            f.write(f"Feature dimension: {feat_dim}\n")
            f.write(f"Clusters: {len(unique_clusters)}\n")
            f.write(f"Hyperedges: {len(hyperedges)}\n")
            f.write(f"Elapsed time: {elapsed_time:.2f}s\n")
            f.write("Mode: GPU-accelerated\n\n")
            f.write("Cluster statistics:\n")
            f.write(f"{'-'*60}\n")
            for cluster_id in sorted(unique_clusters):
                count = np.sum(assignments == cluster_id)
                percentage = count / n_patches * 100
                f.write(f"Cluster {cluster_id}: {count} patches ({percentage:.2f}%)\n")

        return {
            'wsi_name': wsi_name,
            'status': 'success',
            'n_patches': n_patches,
            'n_clusters': len(unique_clusters),
            'time': elapsed_time
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            'wsi_name': wsi_name,
            'status': 'error',
            'error': str(e),
            'time': elapsed_time
        }


def batch_process_dataset_gpu(feature_dir, output_base_dir, dataset_name,
                               display_name=None,
                               n_components=10, concentration=0.1, max_iter=100,
                               skip_existing=True, device=None):
    """Process one dataset in batch mode and save PASG hypergraph outputs."""
    display_name = display_name or dataset_name
    print(f"\n{'='*70}")
    print(f"GPU batch processing dataset: {display_name}")
    print(f"{'='*70}")
    print(f"Feature directory: {feature_dir}")
    print(f"Output directory: {output_base_dir}")
    print(f"Parameters: n_components={n_components}, concentration={concentration}, max_iter={max_iter}")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")
    print(f"Skip existing outputs: {'yes' if skip_existing else 'no'}")

    file_paths = discover_feature_files(feature_dir)

    if len(file_paths) == 0:
        print("Warning: the file list is empty.")
        return None

    dataset_output_dir = os.path.join(output_base_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)

    if skip_existing:
        existing_count = 0
        for feat_file in file_paths:
            basename = os.path.basename(feat_file)
            wsi_name = os.path.splitext(basename)[0]
            wsi_output_dir = os.path.join(dataset_output_dir, wsi_name)

            if os.path.exists(wsi_output_dir):
                required_files = [
                    'cluster_assignments.npy',
                    'cluster_centroids.npy',
                    'hyperedges.pkl',
                    'report.txt'
                ]
                if all(os.path.exists(os.path.join(wsi_output_dir, f)) for f in required_files):
                    existing_count += 1

        print(f"Already processed: {existing_count}/{len(file_paths)}")
        print(f"Remaining files: {len(file_paths) - existing_count}")

    print(f"\nStarting {len(file_paths)} WSI files...")

    results = []
    success_count = 0
    error_count = 0
    skipped_count = 0

    for feat_file in tqdm(file_paths, desc=f"GPU {dataset_name}"):
        result = process_single_wsi_gpu(
            feat_file=feat_file,
            output_dir=dataset_output_dir,
            n_components=n_components,
            concentration=concentration,
            max_iter=max_iter,
            skip_existing=skip_existing,
            device=device
        )

        results.append(result)

        if result['status'] == 'success':
            success_count += 1
        elif result['status'] == 'skipped':
            skipped_count += 1
        else:
            error_count += 1
            print(f"\nError: {result['wsi_name']} - {result.get('error', 'Unknown error')}")

    print(f"\n{'='*70}")
    print(f"{display_name} processing finished")
    print(f"{'='*70}")
    print(f"Success: {success_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Failed: {error_count}")

    newly_processed = success_count
    if newly_processed > 0:
        print(f"Success rate for newly processed files: {newly_processed/(len(file_paths)-skipped_count)*100:.1f}%")
    print(f"Overall completion rate: {(success_count+skipped_count)/len(file_paths)*100:.1f}%")

    success_results = [r for r in results if r['status'] == 'success']
    if success_results:
        times = [r['time'] for r in success_results]
        print("\nTiming:")
        print(f"  Total: {sum(times):.1f}s ({sum(times)/60:.1f} min)")
        print(f"  Average: {np.mean(times):.2f}s/WSI")
        print(f"  Fastest: {min(times):.2f}s")
        print(f"  Slowest: {max(times):.2f}s")

    if success_results:
        n_patches_list = [r['n_patches'] for r in success_results]
        n_clusters_list = [r['n_clusters'] for r in success_results]

        print("\nPatch statistics:")
        print(f"  Total patches: {sum(n_patches_list):,}")
        print(f"  Average: {np.mean(n_patches_list):.0f}")

        print("\nCluster statistics:")
        print(f"  Average number of clusters: {np.mean(n_clusters_list):.2f}")
        print(f"  Cluster range: {min(n_clusters_list)} - {max(n_clusters_list)}")

    report_file = os.path.join(dataset_output_dir, f'{dataset_name}_batch_report_pasg.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"{'='*70}\n")
        f.write(f"{display_name} PASG Batch Processing Report\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total WSI files: {len(file_paths)}\n")
        f.write(f"Success: {success_count}\n")
        f.write(f"Skipped: {skipped_count}\n")
        f.write(f"Failed: {error_count}\n")
        f.write(f"Device: {device}\n\n")

        f.write("Parameters:\n")
        f.write(f"  n_components: {n_components}\n")
        f.write(f"  concentration: {concentration}\n")
        f.write(f"  max_iter: {max_iter}\n\n")

        f.write("Results:\n")
        f.write(f"{'-'*70}\n")

        for result in results:
            if result['status'] == 'success':
                f.write(f"✓ {result['wsi_name']}: "
                       f"{result['n_patches']} patches, "
                       f"{result['n_clusters']} clusters, "
                       f"{result['time']:.2f}s\n")
            elif result['status'] == 'skipped':
                f.write(f"⊙ {result['wsi_name']}: {result.get('message', 'Skipped')}\n")
            else:
                f.write(f"✗ {result['wsi_name']}: {result.get('error', 'Unknown error')}\n")

    print(f"\nSaved batch report: {report_file}")

    return {
        'dataset_name': dataset_name,
        'results': results,
        'success_count': success_count,
        'skipped_count': skipped_count,
        'error_count': error_count
    }


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description="Build PASG hypergraphs with GPU-accelerated Bayesian GMM clustering.")
    parser.add_argument('--dataset', type=str, default='STAD',
                        choices=['STAD', 'THCA', 'CHOL', 'LIHC'],
                        help='Dataset name.')
    parser.add_argument('--base_dir', type=str, default=PROJECT_ROOT,
                        help='Project root directory.')
    parser.add_argument('--n_components', type=int, default=10,
                        help='Maximum number of clusters.')
    parser.add_argument('--concentration', type=float, default=0.1,
                        help='Dirichlet-process concentration parameter.')
    parser.add_argument('--max_iter', type=int, default=100,
                        help='Maximum number of EM iterations.')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use, for example auto, cpu, cuda, or cuda:0.')
    parser.add_argument('--skip_existing', dest='skip_existing', action='store_true', default=True,
                        help='Skip already generated outputs.')
    parser.add_argument('--force', dest='skip_existing', action='store_false',
                        help='Rebuild outputs even if they already exist.')
    args = parser.parse_args()

    print("=" * 70)
    print(" " * 13 + "GPU-Accelerated PASG Hypergraph Construction")
    print("=" * 70)

    n_components = args.n_components
    concentration = args.concentration
    max_iter = args.max_iter
    skip_existing = args.skip_existing
    dataset_key = args.dataset.upper()
    dataset_label = f'TCGA-{dataset_key}'
    base_dir = os.path.abspath(args.base_dir)
    device = resolve_device(args.device)

    print("\nConfiguration:")
    print(f"  Dataset: {dataset_label}")
    print(f"  Max clusters (n_components): {n_components}")
    print(f"  PASG concentration: {concentration}")
    print(f"  Max iterations: {max_iter}")
    print(f"  Skip existing outputs: {'yes' if skip_existing else 'no'}")

    if device.type == 'cuda':
        gpu_index = device.index if device.index is not None else torch.cuda.current_device()
        print(f"  GPU: {torch.cuda.get_device_name(gpu_index)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(gpu_index).total_memory / 1024**3:.1f} GB")
    else:
        print(f"  Device: {device}")

    feature_dir = os.path.join(base_dir, f'WSI_data/{dataset_label}/patch_ft')
    if not os.path.isdir(feature_dir):
        raise FileNotFoundError(f"Feature directory not found: {feature_dir}")

    output_base_dir = os.path.join(base_dir, 'PASG_Hyperedge')
    os.makedirs(output_base_dir, exist_ok=True)

    overall_start_time = time.time()
    result = batch_process_dataset_gpu(
        feature_dir=feature_dir,
        output_base_dir=output_base_dir,
        dataset_name=dataset_key,
        display_name=dataset_label,
        n_components=n_components,
        concentration=concentration,
        max_iter=max_iter,
        skip_existing=skip_existing,
        device=device
    )

    overall_time = time.time() - overall_start_time

    print(f"\n{'='*70}")
    print("Overall summary")
    print(f"{'='*70}")

    if result is None:
        print("Processing failed; no output was generated.")
        return

    total_success = result['success_count']
    total_skipped = result['skipped_count']
    total_error = result['error_count']
    total_wsi = total_success + total_skipped + total_error

    print("Datasets processed: 1")
    print(f"Total WSI files: {total_wsi}")
    print(f"Success: {total_success}")
    print(f"Skipped: {total_skipped}")
    print(f"Failed: {total_error}")
    print(f"Completion rate: {(total_success+total_skipped)/total_wsi*100:.1f}%" if total_wsi else "Completion rate: 0.0%")
    print(f"Total time: {overall_time:.1f}s ({overall_time/60:.1f} min)")

    if total_success > 0:
        print(f"Average speed: {overall_time/total_success:.2f}s/WSI (newly processed files only)")

    summary_file = os.path.join(output_base_dir, 'batch_summary_gpu.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("GPU-Accelerated Dirichlet-Process Clustering - Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {overall_time:.1f}s ({overall_time/60:.1f} min)\n\n")

        f.write("Parameters:\n")
        f.write(f"  n_components: {n_components}\n")
        f.write(f"  concentration: {concentration}\n")
        f.write(f"  max_iter: {max_iter}\n\n")

        f.write("Results:\n")
        f.write(f"  Total WSI files: {total_wsi}\n")
        f.write(f"  Success: {total_success}\n")
        f.write(f"  Skipped: {total_skipped}\n")
        f.write(f"  Failed: {total_error}\n")
        completion = (total_success + total_skipped) / total_wsi * 100 if total_wsi else 0.0
        f.write(f"  Completion rate: {completion:.1f}%\n\n")

        f.write(f"\nDataset: {result['dataset_name']}\n")
        f.write(f"{'-'*70}\n")
        f.write(f"  Success: {result['success_count']}\n")
        f.write(f"  Skipped: {result['skipped_count']}\n")
        f.write(f"  Failed: {result['error_count']}\n")

    print(f"\nSaved summary report: {summary_file}")

    print(f"\n{'='*70}")
    print(" " * 18 + "GPU batch processing finished")
    print(f"{'='*70}")
    print(f"\nOutputs saved under: {output_base_dir}\n")


if __name__ == "__main__":
    main()
