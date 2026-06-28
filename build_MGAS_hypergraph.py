#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build MGAS graphs and hypergraphs, optionally with GPU acceleration."""

import os
import numpy as np
import torch
from torch_geometric.data import Data
from pathlib import Path
import networkx as nx
import math
from collections import deque
import pickle
from tqdm import tqdm
import time
import argparse

MAX_SPLIT_DEPTH = 2

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


def discover_sample_pairs(coor_dir, ft_dir):
    """Discover samples that have both coordinate and feature files."""
    if not os.path.isdir(coor_dir):
        raise FileNotFoundError(f"Coordinate directory not found: {coor_dir}")
    if not os.path.isdir(ft_dir):
        raise FileNotFoundError(f"Feature directory not found: {ft_dir}")

    coor_map = {}
    for name in sorted(os.listdir(coor_dir)):
        if name.endswith("_coors.npy"):
            sample_id = name[:-10]
            coor_map[sample_id] = os.path.join(coor_dir, name)

    ft_map = {}
    for name in sorted(os.listdir(ft_dir)):
        if name.endswith("_fts.npy"):
            sample_id = name[:-8]
            ft_map[sample_id] = os.path.join(ft_dir, name)

    sample_ids = sorted(set(coor_map) & set(ft_map))
    return [(sample_id, coor_map[sample_id], ft_map[sample_id]) for sample_id in sample_ids]

def build_graph_gpu(coordinates, features, spatial_threshold=2.9, cosine_threshold=0.75, device=None):
    """Build a graph from coordinates and features using chunked pairwise processing."""
    device = resolve_device(device)
    num_nodes = coordinates.shape[0]

    coord_tensor = torch.from_numpy(coordinates).float().to(device)
    feat_tensor = torch.from_numpy(features).float().to(device)

    feat_norm = torch.nn.functional.normalize(feat_tensor, p=2, dim=1)

    edge_dict = {i: set() for i in range(num_nodes)}

    chunk_size = 2000 if device.type == 'cuda' else 1000

    for i in range(0, num_nodes, chunk_size):
        end_i = min(i + chunk_size, num_nodes)

        coord_chunk = coord_tensor[i:end_i]
        dist_matrix = torch.cdist(coord_chunk, coord_tensor, p=2)
        spatial_neighbors = (dist_matrix <= spatial_threshold) & (dist_matrix > 0)

        for local_idx in range(end_i - i):
            global_idx = i + local_idx
            neighbor_mask = spatial_neighbors[local_idx]
            neighbor_indices = torch.where(neighbor_mask)[0]

            if len(neighbor_indices) > 0:
                feat_i = feat_norm[global_idx:global_idx+1]
                feat_neighbors = feat_norm[neighbor_indices]
                cosine_sims = torch.mm(feat_i, feat_neighbors.t())[0]
                valid_mask = cosine_sims >= cosine_threshold
                valid_neighbors = neighbor_indices[valid_mask].cpu().numpy()
                edge_dict[global_idx].update(valid_neighbors.tolist())

    edge_list_from = []
    edge_list_to = []
    node_degrees = np.zeros(num_nodes, dtype=np.int32)

    for node_id, neighbors in edge_dict.items():
        node_degrees[node_id] = len(neighbors)
        for neighbor_id in neighbors:
            edge_list_from.append(node_id)
            edge_list_to.append(neighbor_id)

    edge_index = torch.tensor([edge_list_from, edge_list_to], dtype=torch.long)
    return edge_index, node_degrees

def generate_hypergraph_from_graph(nx_graph):
    """Generate hyperedges from a NetworkX graph without reusing the original edges."""
    if nx.is_connected(nx_graph):
        hyperedges = create_generalized_ball_graph(nx_graph)
    else:
        connected_components = list(nx.connected_components(nx_graph))
        connected_subgraphs = [nx_graph.subgraph(component) for component in connected_components]
        hyperedges = []

        for connected_subgraph in connected_subgraphs:
            hyperedge = create_generalized_ball_graph(connected_subgraph)
            hyperedges.extend(hyperedge)

    return hyperedges

def create_generalized_ball_graph(nx_graph):
    """Create generalized-ball hyperedges from the input graph."""
    initial_ball_count = math.isqrt(len(nx_graph))

    if initial_ball_count == 1:
        initial_balls = [nx_graph]
    else:
        initial_balls = initialize_generalized_ball_graph(nx_graph, initial_ball_count)

    generalized_balls = []
    for initial_ball in initial_balls:
        split_balls = []
        recursively_split_ball(initial_ball, split_balls, current_depth=0)
        generalized_balls.extend(split_balls)

    return [tuple(ball.nodes()) for ball in generalized_balls] if len(generalized_balls) > 1 else []

def initialize_generalized_ball_graph(nx_graph, initial_ball_count):
    """Initialize generalized balls from high-degree center nodes."""
    degree_dict = dict(nx_graph.degree())
    sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)
    center_nodes = sorted_nodes[:initial_ball_count]
    center_nodes_dict = assign_nodes_to_multiple_centers(nx_graph, center_nodes)
    return [nx.subgraph(nx_graph, cluster) for cluster in center_nodes_dict.values()]

def assign_nodes_to_multiple_centers(G, centers):
    """Assign graph nodes to multiple centers with multi-source BFS."""
    center_nodes_dict = {center: set() for center in centers}
    center_queues = {center: deque([center]) for center in centers}
    visited_nodes = {center: center for center in centers}

    while any(center_queues.values()):
        for center in centers:
            if center_queues[center]:
                current_node = center_queues[center].popleft()
                center_nodes_dict[center].add(current_node)

                for neighbor in G.neighbors(current_node):
                    if neighbor not in visited_nodes:
                        visited_nodes[neighbor] = center
                        center_queues[center].append(neighbor)

    return center_nodes_dict

def recursively_split_ball(nx_graph, split_balls, current_depth):
    """Recursively split a graph into smaller generalized balls."""
    if len(nx_graph) == 1:
        return

    degree_dict = dict(nx_graph.degree())
    sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)
    center_nodes = sorted_nodes[:2]
    center_nodes_dict = assign_nodes_to_multiple_centers(nx_graph, center_nodes)
    clusters = [cluster for cluster in center_nodes_dict.values()]

    cluster_a, cluster_b = clusters[0], clusters[1]
    subgraph_a, subgraph_b = nx.subgraph(nx_graph, cluster_a), nx.subgraph(nx_graph, cluster_b)

    if len(subgraph_a.edges()) == 0 or len(subgraph_b.edges()) == 0:
        split_balls.append(nx_graph)
    else:
        avg_degree = nx_graph.number_of_edges() / len(nx_graph)
        avg_degree_a = subgraph_a.number_of_edges() / len(subgraph_a)
        avg_degree_b = subgraph_b.number_of_edges() / len(subgraph_b)
        if avg_degree < avg_degree_a + avg_degree_b and current_depth < MAX_SPLIT_DEPTH - 1:
            recursively_split_ball(subgraph_a, split_balls, current_depth + 1)
            recursively_split_ball(subgraph_b, split_balls, current_depth + 1)
        else:
            split_balls.append(nx_graph)

def process_sample(coor_file, ft_file, save_dir, spatial_threshold=2.9, cosine_threshold=0.75, patch_size=256,
                   device=None):
    """Process one sample: load data, build the graph, and save outputs."""
    try:
        sample_id = Path(coor_file).stem.replace('_coors', '')
        sample_dir = os.path.join(save_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)
        coordinates = np.load(coor_file)
        features = np.load(ft_file)

        if len(features.shape) == 1:
            feat_dim = 512
            num_patches = features.shape[0] // feat_dim
            if features.shape[0] % feat_dim != 0:
                print(f"  Warning: Feature size {features.shape[0]} not divisible by {feat_dim}")
                return None
            features = features.reshape(num_patches, feat_dim)

        coordinates = coordinates.astype(np.float32) / patch_size
        if coordinates.shape[0] != features.shape[0]:
            print(f"  Dimension mismatch: coordinates {coordinates.shape[0]} != features {features.shape[0]}")
            return None

        num_nodes = coordinates.shape[0]
        edge_index, node_degrees = build_graph_gpu(
            coordinates,
            features,
            spatial_threshold,
            cosine_threshold,
            device=device,
        )

        edge_path = os.path.join(sample_dir, "edges.pt")
        torch.save(edge_index, edge_path)
        degree_path = os.path.join(sample_dir, "degrees.npy")
        np.save(degree_path, node_degrees)

        num_edges = edge_index.shape[1]

        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(range(num_nodes))
        edge_list = edge_index.t().numpy()
        nx_graph.add_edges_from(edge_list)

        hyperedges = generate_hypergraph_from_graph(nx_graph)
        hyperedge_path = os.path.join(sample_dir, "hyperedges.pkl")
        with open(hyperedge_path, 'wb') as f:
            pickle.dump(hyperedges, f)

        stats_path = os.path.join(sample_dir, "report.txt")
        sqrt_vertices = int(np.sqrt(num_nodes))
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Sample: {sample_id}\n")
            f.write("=" * 80 + "\n\n")

            f.write("Original Graph:\n")
            f.write(f"  Vertices: {num_nodes}\n")
            f.write(f"  Edges: {num_edges}\n")
            f.write(f"  Device: {resolve_device(device)}\n")
            f.write(f"  Average degree: {np.mean(node_degrees):.2f}\n")
            f.write(f"  Max degree: {np.max(node_degrees)}\n\n")

            f.write("MGAS Hypergraph:\n")
            f.write(f"  Vertices: {num_nodes}\n")
            f.write(f"  sqrt(Vertices): {sqrt_vertices}\n")
            f.write(f"  Hyperedges: {len(hyperedges)}\n")
            f.write(f"  Hyperedges/sqrt(Vertices): {len(hyperedges)/sqrt_vertices:.2f}\n\n")

            f.write("Node Degree Distribution:\n")
            f.write(f"  Mean: {np.mean(node_degrees):.2f}\n")
            f.write(f"  Median: {np.median(node_degrees):.2f}\n")
            f.write(f"  Std: {np.std(node_degrees):.2f}\n")
            f.write(f"  Min: {np.min(node_degrees)}\n")
            f.write(f"  Max: {np.max(node_degrees)}\n\n")

            isolated = np.sum(node_degrees == 0)
            f.write(f"Isolated nodes: {isolated} ({isolated/len(node_degrees)*100:.2f}%)\n")
            f.write("=" * 80 + "\n")

        return {
            'sample_id': sample_id,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'num_hyperedges': len(hyperedges),
            'avg_degree': np.mean(node_degrees),
            'success': True
        }

    except Exception as e:
        print(f"Error processing {coor_file}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Build MGAS hypergraphs in batch.")
    parser.add_argument('--dataset', type=str, default='STAD',
                        choices=['STAD', 'THCA', 'CHOL', 'LIHC'],
                        help='Dataset name.')
    parser.add_argument('--base_dir', type=str, default=PROJECT_ROOT,
                        help='Project root directory.')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use, for example auto, cpu, cuda, or cuda:0.')
    args = parser.parse_args()

    dataset_name = args.dataset.upper()
    device = resolve_device(args.device)

    base_dir = os.path.abspath(args.base_dir)
    coor_dir = os.path.join(base_dir, f'WSI_data/TCGA-{dataset_name}/patch_coor')
    ft_dir = os.path.join(base_dir, f'WSI_data/TCGA-{dataset_name}/patch_ft')
    save_dir = os.path.join(base_dir, f'MGAS_Hyperedge/{dataset_name}')

    os.makedirs(save_dir, exist_ok=True)

    print("=" * 100)
    print(f"Processing TCGA-{dataset_name} in batch")
    print("=" * 100)
    print(f"Coordinate directory: {coor_dir}")
    print(f"Feature directory: {ft_dir}")
    if device.type == 'cuda':
        gpu_index = device.index if device.index is not None else torch.cuda.current_device()
        print(f"Using device: {torch.cuda.get_device_name(gpu_index)} ({device})")
    else:
        print(f"Using device: {device}")

    sample_pairs = discover_sample_pairs(coor_dir, ft_dir)

    print(f"Loaded {len(sample_pairs)} samples from patch_coor/patch_ft")
    print(f"Output directory: {save_dir}")
    print("=" * 100)

    results = []
    failed_samples = []
    start_time = time.time()

    for sample_id, coor_path, ft_path in tqdm(sample_pairs, desc="Processing samples"):
        result = process_sample(coor_path, ft_path, save_dir,
                               spatial_threshold=2.9,
                               cosine_threshold=0.75,
                               patch_size=256,
                               device=device)

        if result is not None:
            results.append(result)
        else:
            failed_samples.append((sample_id, "Processing failed"))

    elapsed_time = time.time() - start_time

    print("\n" + "=" * 100)
    print("Processing finished")
    print("=" * 100)
    print(f"Elapsed time: {elapsed_time/60:.2f} minutes")
    print(f"Successful samples: {len(results)}")
    print(f"Failed samples: {len(failed_samples)}")

    if len(results) > 0:
        print("\nSummary:")
        total_nodes = sum(r['num_nodes'] for r in results)
        total_edges = sum(r['num_edges'] for r in results)
        total_hyperedges = sum(r['num_hyperedges'] for r in results)
        avg_degree = np.mean([r['avg_degree'] for r in results])

        print(f"  Total nodes: {total_nodes:,}")
        print(f"  Total edges: {total_edges:,}")
        print(f"  Total hyperedges: {total_hyperedges:,}")
        print(f"  Average node degree: {avg_degree:.2f}")

    if len(failed_samples) > 0:
        print("\nFailed samples:")
        for sample, reason in failed_samples[:10]:
            print(f"  - {sample}: {reason}")
        if len(failed_samples) > 10:
            print(f"  ... and {len(failed_samples)-10} more")

    print("=" * 100)

    summary_path = os.path.join(save_dir, "summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write(f"TCGA-{dataset_name} MGAS Hypergraph Summary\n")
        f.write("=" * 100 + "\n")
        f.write(f"Total samples: {len(sample_pairs)}\n")
        f.write(f"Successful: {len(results)}\n")
        f.write(f"Failed: {len(failed_samples)}\n")
        f.write(f"Processing time: {elapsed_time/60:.2f} minutes\n\n")

        if len(results) > 0:
            f.write("Statistics:\n")
            f.write(f"  Total nodes: {total_nodes:,}\n")
            f.write(f"  Total edges: {total_edges:,}\n")
            f.write(f"  Total hyperedges: {total_hyperedges:,}\n")
            f.write(f"  Average node degree: {avg_degree:.2f}\n")

        f.write("=" * 100 + "\n")

    print(f"Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
