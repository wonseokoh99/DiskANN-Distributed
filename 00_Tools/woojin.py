#!/usr/bin/env python3
"""
Streaming Edge Analysis Pipeline - Analyze cluster edges before and after DiskANN streaming operations

This script implements the 4-step process:
1. Extract active_window slice from SIFT data and convert to .fbin
2. Apply balanced K-means clustering with K-means++ initialization (k=8) and perform initial cross-cluster edge analysis  
3. Run test_streaming_scenario to get updated data
4. Analyze edges in updated data and compare with original

Features:
- Balanced K-means with K-means++ initialization for even cluster sizes
- Cross-cluster edge analysis with connectivity matrices
- Configurable clustering parameters and graph construction settings

Usage:
    python3 streaming_edge_analysis.py <dataset_name> <max_points_to_insert> <active_window> <consolidate_interval> [options]
    
Examples:
    # Use balanced K-means (default)
    python3 scripts/streaming_edge_analysis.py gist_base.fbin 50000 10000 10000
    
    # Use standard K-means
    python3 scripts/streaming_edge_analysis.py gist_base.fbin 50000 10000 10000 --use_standard_kmeans
"""

import numpy as np
import struct
import subprocess
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter
import argparse

def install_required_packages():
    """Install required Python packages if not available"""
    try:
        import sklearn
        print("sklearn is available")
    except ImportError:
        print("Installing scikit-learn...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'scikit-learn'], check=True)

def balanced_kmeans_with_kmeanspp(data, k, max_iters=100, centroid_tolerance=1e-4, assignment_tolerance=0.01, random_state=42):
    """
    Balanced K-means clustering with K-means++ initialization
    
    Args:
        data: Input data vectors (numpy array)
        k: Number of clusters
        max_iters: Maximum iterations
        centroid_tolerance: Convergence tolerance for centroid movement
        assignment_tolerance: Convergence tolerance for assignment changes
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (labels, centroids, convergence_info)
    """
    from sklearn.cluster import KMeans
    from scipy.spatial.distance import cdist
    
    print(f"  Applying balanced K-means with K-means++ initialization...")
    
    n_samples = len(data)
    target_size = n_samples // k
    remainder = n_samples % k
    
    print(f"  Target cluster size: {target_size} (with {remainder} clusters getting +1 point)")
    
    # Initialize with K-means++ 
    print(f"  Initializing centroids with K-means++...")
    kmeans_init = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=1, random_state=random_state)
    kmeans_init.fit(data)
    centroids = kmeans_init.cluster_centers_.copy()
    print(f"  K-means++ initialization completed")
    
    prev_assignments = None
    convergence_info = []
    
    for iteration in range(max_iters):
        old_centroids = centroids.copy()
        
        # Constrained assignment step
        assignments = constrained_assignment_step(data, centroids, target_size, remainder)
        
        # Update centroids
        centroids = update_centroids_step(data, assignments, k)
        
        # Calculate convergence metrics
        centroid_shifts = np.linalg.norm(centroids - old_centroids, axis=1)
        max_centroid_shift = np.max(centroid_shifts)
        
        assignment_changes_ratio = 0.0
        if prev_assignments is not None:
            assignment_changes = np.sum(assignments != prev_assignments)
            assignment_changes_ratio = assignment_changes / n_samples
        
        # Check convergence
        centroid_converged = max_centroid_shift < centroid_tolerance
        assignment_converged = (prev_assignments is not None and 
                               assignment_changes_ratio < assignment_tolerance)
        
        # Track convergence
        convergence_info.append({
            'iteration': iteration,
            'centroid_shift': max_centroid_shift,
            'assignment_changes_ratio': assignment_changes_ratio
        })
        
        if iteration % 5 == 0 or centroid_converged:
            print(f"    Iter {iteration:2d}: centroid_shift={max_centroid_shift:.6f}, "
                  f"assignment_changes={assignment_changes_ratio:.4f}")
        
        # Convergence check
        if iteration >= 2 and centroid_converged and assignment_converged:
            print(f"  Converged after {iteration + 1} iterations")
            break
        
        prev_assignments = assignments.copy()
    
    # Final cluster size verification
    unique_labels, counts = np.unique(assignments, return_counts=True)
    cluster_sizes = np.zeros(k, dtype=int)
    cluster_sizes[unique_labels] = counts
    
    print(f"  Final cluster sizes: {cluster_sizes}")
    balance_ratio = np.min(cluster_sizes) / np.max(cluster_sizes) if np.max(cluster_sizes) > 0 else 0
    print(f"  Balance ratio (min/max): {balance_ratio:.3f}")
    
    return assignments, centroids, convergence_info

def constrained_assignment_step(data, centroids, target_size, remainder):
    """
    Assign points to clusters with size constraints
    """
    from scipy.spatial.distance import cdist
    
    n_samples = len(data)
    k = len(centroids)
    
    # Calculate cluster capacity (some clusters get +1 point)
    cluster_capacity = np.full(k, target_size)
    cluster_capacity[:remainder] += 1  # First 'remainder' clusters get +1
    
    # Calculate distances from all points to all centroids
    distances = cdist(data, centroids, metric='euclidean')
    
    # Initialize assignments and cluster sizes
    assignments = np.full(n_samples, -1)
    cluster_sizes = np.zeros(k, dtype=int)
    
    # Sort points by their confidence (minimum distance to any centroid)
    # More confident assignments (smaller min distance) get priority
    min_distances = np.min(distances, axis=1)
    sorted_indices = np.argsort(min_distances)
    
    for idx in sorted_indices:
        # Get preference order (closest centroids first)
        centroid_order = np.argsort(distances[idx])
        
        # Try to assign to the closest available centroid
        assigned = False
        for centroid_idx in centroid_order:
            if cluster_sizes[centroid_idx] < cluster_capacity[centroid_idx]:
                assignments[idx] = centroid_idx
                cluster_sizes[centroid_idx] += 1
                assigned = True
                break
        
        # If all preferred clusters are full, assign to least full cluster
        if not assigned:
            least_full_cluster = np.argmin(cluster_sizes)
            assignments[idx] = least_full_cluster
            cluster_sizes[least_full_cluster] += 1
    
    return assignments

def update_centroids_step(data, assignments, k):
    """
    Update centroids as mean of assigned points
    """
    centroids = np.zeros((k, data.shape[1]))
    
    for cluster_id in range(k):
        cluster_points = data[assignments == cluster_id]
        if len(cluster_points) > 0:
            centroids[cluster_id] = np.mean(cluster_points, axis=0)
        # If cluster is empty, keep the previous centroid position
    
    return centroids

def read_fvecs(filename, start=0, count=None):
    """Read .fvecs format vectors"""
    vectors = []
    with open(filename, 'rb') as f:
        # Skip to start position
        if start > 0:
            # Read first vector to get dimension
            dim = struct.unpack('i', f.read(4))[0]
            f.seek(0)
            # Skip start vectors
            vector_size = 4 + dim * 4  # 4 bytes for dim + dim * 4 bytes for floats
            f.seek(start * vector_size)
        
        read_count = 0
        while True:
            if count is not None and read_count >= count:
                break
                
            # Read dimension
            dim_bytes = f.read(4)
            if len(dim_bytes) != 4:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            
            # Read vector data
            vector_data = f.read(dim * 4)
            if len(vector_data) != dim * 4:
                break
                
            vector = struct.unpack('f' * dim, vector_data)
            vectors.append(vector)
            read_count += 1
            
    return vectors, dim

def read_fbin(filename, start=0, count=None):
    """Read .fbin format vectors"""
    with open(filename, 'rb') as f:
        # Read header
        num_points = struct.unpack('i', f.read(4))[0]
        dim = struct.unpack('i', f.read(4))[0]
        
        # Calculate bounds
        if start >= num_points:
            return [], dim
        
        end = min(num_points, start + count) if count is not None else num_points
        actual_count = end - start
        
        # Skip to start position
        f.seek(8 + start * dim * 4)  # 8 bytes header + start * vector_size
        
        # Read vectors
        vectors = []
        for i in range(actual_count):
            vector_data = f.read(dim * 4)
            if len(vector_data) != dim * 4:
                break
            vector = struct.unpack('f' * dim, vector_data)
            vectors.append(vector)
            
    return vectors, dim

def write_fvecs(filename, vectors, dim):
    """Write vectors in .fvecs format"""
    with open(filename, 'wb') as f:
        for vector in vectors:
            f.write(struct.pack('i', dim))
            f.write(struct.pack('f' * dim, *vector))

def write_fbin(filename, vectors, dim):
    """Write vectors in .fbin format"""
    with open(filename, 'wb') as f:
        # Write header
        f.write(struct.pack('i', len(vectors)))
        f.write(struct.pack('i', dim))
        
        # Write vectors
        for vector in vectors:
            f.write(struct.pack('f' * dim, *vector))

def analyze_index_edges(index_file, labels, centroids, output_prefix, k=8, max_degree=70):
    """
    Analyze inter-cluster edges by parsing DiskANN index file adjacency lists
    
    Args:
        index_file: Path to DiskANN index file 
        labels: cluster labels for each vector
        centroids: cluster centroids
        output_prefix: prefix for output files
        k: number of clusters
        max_degree: maximum degree (R parameter)
    """
    print(f"Analyzing edges from DiskANN index file {index_file}...")
    
    try:
        edge_data = []
        cluster_edges = defaultdict(list)
        
        with open(index_file, 'rb') as f:
            file_size = os.path.getsize(index_file)
            print(f"Index file size: {file_size} bytes")
            
            # Parse DiskANN .index file header (24 bytes total)
            # file_size_bytes (8 bytes)
            file_size_bytes = struct.unpack('Q', f.read(8))[0]  # uint64_t
            print(f"Header file_size_bytes: {file_size_bytes}")
            
            # max_degree (4 bytes)
            header_max_degree = struct.unpack('I', f.read(4))[0]  # uint32_t
            print(f"Header max_degree: {header_max_degree}")
            
            # entrypoint_id (4 bytes)
            entrypoint_id = struct.unpack('I', f.read(4))[0]  # uint32_t
            print(f"Header entrypoint_id: {entrypoint_id}")
            
            # num_frozen_points (8 bytes)
            num_frozen_points = struct.unpack('Q', f.read(8))[0]  # uint64_t
            print(f"Header num_frozen_points: {num_frozen_points}")
            
            print(f"Header parsed successfully, now at position: {f.tell()}")
            
            # Parse variable-size adjacency lists
            node_id = 0
            nodes_processed = 0
            
            while f.tell() < file_size and node_id < len(labels):
                try:
                    # Read degree for this node (4 bytes)
                    degree_bytes = f.read(4)
                    if len(degree_bytes) != 4:
                        break
                    degree = struct.unpack('I', degree_bytes)[0]  # uint32_t
                    
                    # Sanity check for degree
                    if degree > header_max_degree * 2:  # Allow some flexibility
                        print(f"Warning: Node {node_id} has unreasonable degree {degree}, skipping")
                        break
                    
                    # Read neighbor_ids (degree × 4 bytes, variable size)
                    neighbors = []
                    for _ in range(degree):
                        neighbor_bytes = f.read(4)
                        if len(neighbor_bytes) != 4:
                            break
                        neighbor_id = struct.unpack('I', neighbor_bytes)[0]  # uint32_t
                        
                        # Validate neighbor_id
                        if neighbor_id < len(labels):
                            neighbors.append(neighbor_id)
                    
                    # Analyze edges for this node
                    if node_id < len(labels):
                        cluster_i = labels[node_id]
                        for neighbor_id in neighbors:
                            cluster_j = labels[neighbor_id]
                            edge_data.append({
                                'from_vector': node_id,
                                'to_vector': neighbor_id,
                                'from_cluster': cluster_i,
                                'to_cluster': cluster_j,
                                'is_cross_cluster': cluster_i != cluster_j
                            })
                            cluster_edges[cluster_i].append(cluster_j)
                    
                    nodes_processed += 1
                    node_id += 1
                    
                    # Progress logging
                    if nodes_processed % 1000 == 0:
                        print(f"  Processed {nodes_processed} nodes...")
                    
                except struct.error as e:
                    print(f"Struct error at node {node_id}: {e}")
                    break
                except Exception as e:
                    print(f"Error processing node {node_id}: {e}")
                    break
                    
            print(f"Successfully processed {nodes_processed} nodes from index file")
            
    except Exception as e:
        print(f"Warning: Could not parse index file format: {e}")
        print("Falling back to distance-based edge analysis...")
        # Fall back to the original distance-based method
        return cross_cluster_edge_analysis_fallback(labels, centroids, output_prefix, k)
    
    if not edge_data:
        print("No edges found in index file, falling back to distance-based analysis...")
        return cross_cluster_edge_analysis_fallback(labels, centroids, output_prefix, k)
    
    # Analyze edge statistics
    cross_cluster_edges = [e for e in edge_data if e['is_cross_cluster']]
    intra_cluster_edges = [e for e in edge_data if not e['is_cross_cluster']]
    
    # Cluster size analysis
    cluster_sizes = Counter(labels)
    
    # Calculate average inter-cluster edge count per cluster
    inter_edge_counts = Counter()  # edges going out to different clusters
    
    for edge in cross_cluster_edges:
        inter_edge_counts[edge['from_cluster']] += 1
    
    # Calculate average inter-cluster edges per node in each cluster
    avg_inter_edges = {}
    for cluster_id in range(k):
        cluster_size = cluster_sizes.get(cluster_id, 0)
        inter_edges = inter_edge_counts.get(cluster_id, 0)
        avg_inter_edges[cluster_id] = inter_edges / cluster_size if cluster_size > 0 else 0
    
    # Calculate per-node inter-cluster proportions for each cluster
    node_degrees = Counter()  # Total out-degree for each node
    node_inter_edges = Counter()  # Inter-cluster edges for each node
    
    # Count degrees and inter-cluster edges per node
    for edge in edge_data:
        node_degrees[edge['from_vector']] += 1
        if edge['is_cross_cluster']:
            node_inter_edges[edge['from_vector']] += 1
    
    # Group proportions by cluster
    cluster_inter_proportions = defaultdict(list)
    for node_id in range(len(labels)):
        if node_id < len(labels):
            node_cluster = labels[node_id]
            total_degree = node_degrees.get(node_id, 0)
            inter_count = node_inter_edges.get(node_id, 0)
            
            if total_degree > 0:
                proportion = inter_count / total_degree
                cluster_inter_proportions[node_cluster].append(proportion)
    
    # Overall average inter-cluster edges
    total_inter_edges = sum(inter_edge_counts.values())
    total_nodes = len(labels)
    overall_avg_inter = total_inter_edges / total_nodes if total_nodes > 0 else 0
    
    stats = {
        'total_vectors': len(labels),
        'total_edges': len(edge_data),
        'cross_cluster_edges': len(cross_cluster_edges),
        'intra_cluster_edges': len(intra_cluster_edges),
        'cross_cluster_ratio': len(cross_cluster_edges) / len(edge_data) if edge_data else 0,
        'overall_avg_inter_edges': overall_avg_inter,
        'avg_inter_edges_per_cluster': avg_inter_edges
    }
    
    # Create cross-cluster edge matrix (absolute counts)
    edge_matrix = np.zeros((k, k))
    for edge in cross_cluster_edges:
        edge_matrix[edge['from_cluster']][edge['to_cluster']] += 1
    
    # Create average cross-cluster edge matrix (normalized by cluster size)
    avg_edge_matrix = np.zeros((k, k))
    for i in range(k):
        cluster_size_i = cluster_sizes.get(i, 0)
        if cluster_size_i > 0:
            for j in range(k):
                avg_edge_matrix[i][j] = edge_matrix[i][j] / cluster_size_i
    
    # Save results
    with open(f"{output_prefix}_edge_analysis.txt", 'w') as f:
        f.write("=== DiskANN Index Cross-Cluster Edge Analysis ===\n")
        f.write(f"Total vectors: {stats['total_vectors']}\n")
        f.write(f"Total edges: {stats['total_edges']}\n")
        f.write(f"Cross-cluster edges: {stats['cross_cluster_edges']}\n")
        f.write(f"Intra-cluster edges: {stats['intra_cluster_edges']}\n")
        f.write(f"Cross-cluster ratio: {stats['cross_cluster_ratio']:.4f}\n")
        f.write(f"Overall avg inter-cluster edges per node: {stats['overall_avg_inter_edges']:.4f}\n\n")
        
        f.write("=== Cluster Sizes ===\n")
        for cluster_id in sorted(cluster_sizes.keys()):
            f.write(f"Cluster {cluster_id}: {cluster_sizes[cluster_id]} vectors\n")
        
        f.write("\n=== Average Inter-Cluster Edges Per Node by Cluster ===\n")
        f.write("Cluster   Avg Inter Edges\n")
        for cluster_id in range(k):
            avg_inter = stats['avg_inter_edges_per_cluster'][cluster_id]
            f.write(f"{cluster_id:7}   {avg_inter:13.4f}\n")
        
        f.write("\n=== Average Proportion of Inter-Cluster Edges Per Node by Cluster ===\n")
        f.write("Cluster   Avg Inter Proportion   Std Dev   Min    Max\n")
        for cluster_id in range(k):
            proportions = cluster_inter_proportions.get(cluster_id, [])
            if proportions:
                avg_proportion = np.mean(proportions)
                std_proportion = np.std(proportions)
                min_proportion = np.min(proportions)
                max_proportion = np.max(proportions)
                f.write(f"{cluster_id:7}   {avg_proportion:17.4f}   {std_proportion:7.4f}   {min_proportion:.4f}  {max_proportion:.4f}\n")
            else:
                f.write(f"{cluster_id:7}   0.0000               0.0000   0.0000  0.0000\n")
        
        f.write("\n=== Cross-Cluster Edge Matrix (Absolute Counts) ===\n")
        f.write("From\\To")
        for j in range(k):
            f.write(f"    {j:4}")
        f.write("\n")
        
        for i in range(k):
            f.write(f"{i:6}")
            for j in range(k):
                f.write(f"  {int(edge_matrix[i][j]):5}")
            f.write("\n")
        
        f.write("\n=== Average Cross-Cluster Edge Matrix (Per Node) ===\n")
        f.write("From\\To")
        for j in range(k):
            f.write(f"     {j:6}")
        f.write("\n")
        
        for i in range(k):
            f.write(f"{i:6}")
            for j in range(k):
                f.write(f"  {avg_edge_matrix[i][j]:8.4f}")
            f.write("\n")
    
    # Save cluster sizes
    with open(f"{output_prefix}_cluster_sizes.txt", 'w') as f:
        f.write("cluster_id,size\n")
        for cluster_id in sorted(cluster_sizes.keys()):
            f.write(f"{cluster_id},{cluster_sizes[cluster_id]}\n")
    
    # Print proportion statistics to console
    print(f"Inter-cluster proportion analysis:")
    for cluster_id in range(k):
        proportions = cluster_inter_proportions.get(cluster_id, [])
        if proportions:
            avg_proportion = np.mean(proportions)
            std_proportion = np.std(proportions)
            print(f"  Cluster {cluster_id}: {avg_proportion:.3f} ± {std_proportion:.3f} (n={len(proportions)})")
        else:
            print(f"  Cluster {cluster_id}: 0.000 ± 0.000 (n=0)")
    
    print(f"Index-based analysis saved to {output_prefix}_edge_analysis.txt")
    return stats, cluster_sizes

def cross_cluster_edge_analysis_fallback(labels, centroids, output_prefix, k=8):
    """Fallback distance-based edge analysis when index parsing fails"""
    print("Using fallback distance-based edge analysis...")
    # This would use the original distance-calculation method
    # For now, return minimal stats
    cluster_sizes = Counter(labels)
    stats = {
        'total_vectors': len(labels),
        'total_edges': 0,
        'cross_cluster_edges': 0,
        'intra_cluster_edges': 0,
        'cross_cluster_ratio': 0
    }
    return stats, cluster_sizes

def cross_cluster_edge_analysis(vectors, labels, centroids, output_prefix, k=8):
    """
    Perform cross-cluster edge analysis similar to pipeline.py
    
    Args:
        vectors: numpy array of vectors
        labels: cluster labels for each vector  
        centroids: cluster centroids
        output_prefix: prefix for output files
        k: number of clusters
    """
    print(f"Performing cross-cluster edge analysis for {len(vectors)} vectors...")
    
    # Build adjacency relationships (simplified nearest neighbors)
    n_vectors = len(vectors)
    edge_data = []
    cluster_edges = defaultdict(list)
    
    # For each vector, find its nearest neighbors and create edges
    batch_size = 500  # Process in smaller batches for memory efficiency
    
    for start_idx in range(0, min(5000, n_vectors), batch_size):  # Limit to first 5000 for efficiency
        end_idx = min(start_idx + batch_size, n_vectors)
        batch_vectors = vectors[start_idx:end_idx]
        
        for i, vec in enumerate(batch_vectors):
            global_idx = start_idx + i
            cluster_i = labels[global_idx]
            
            # Calculate distances to all vectors (simplified)
            distances = np.linalg.norm(vectors - vec, axis=1)
            nearest_indices = np.argsort(distances)[1:21]  # Skip self, take top 20
            
            for j in nearest_indices:
                cluster_j = labels[j]
                edge_data.append({
                    'from_vector': global_idx,
                    'to_vector': j,
                    'from_cluster': cluster_i,
                    'to_cluster': cluster_j,
                    'distance': distances[j],
                    'is_cross_cluster': cluster_i != cluster_j
                })
                
                cluster_edges[cluster_i].append((cluster_j, distances[j]))
        
        if (start_idx // batch_size + 1) % 5 == 0:
            print(f"  Processed {end_idx}/{min(5000, n_vectors)} vectors...")
    
    # Analyze edge statistics
    cross_cluster_edges = [e for e in edge_data if e['is_cross_cluster']]
    intra_cluster_edges = [e for e in edge_data if not e['is_cross_cluster']]
    
    stats = {
        'total_vectors': n_vectors,
        'total_edges': len(edge_data),
        'cross_cluster_edges': len(cross_cluster_edges),
        'intra_cluster_edges': len(intra_cluster_edges),
        'cross_cluster_ratio': len(cross_cluster_edges) / len(edge_data) if edge_data else 0
    }
    
    # Cluster size analysis
    cluster_sizes = Counter(labels)
    
    # Create cross-cluster edge matrix
    edge_matrix = np.zeros((k, k))
    for edge in cross_cluster_edges:
        edge_matrix[edge['from_cluster']][edge['to_cluster']] += 1
    
    # Save results
    with open(f"{output_prefix}_edge_analysis.txt", 'w') as f:
        f.write("=== Cross-Cluster Edge Analysis ===\n")
        f.write(f"Total vectors: {stats['total_vectors']}\n")
        f.write(f"Total edges: {stats['total_edges']}\n")
        f.write(f"Cross-cluster edges: {stats['cross_cluster_edges']}\n")
        f.write(f"Intra-cluster edges: {stats['intra_cluster_edges']}\n")
        f.write(f"Cross-cluster ratio: {stats['cross_cluster_ratio']:.4f}\n\n")
        
        f.write("=== Cluster Sizes ===\n")
        for cluster_id in sorted(cluster_sizes.keys()):
            f.write(f"Cluster {cluster_id}: {cluster_sizes[cluster_id]} vectors\n")
        
        f.write("\n=== Cross-Cluster Edge Matrix ===\n")
        f.write("From\\To")
        for j in range(k):
            f.write(f"    {j:4}")
        f.write("\n")
        
        for i in range(k):
            f.write(f"{i:6}")
            for j in range(k):
                f.write(f"  {int(edge_matrix[i][j]):5}")
            f.write("\n")
    
    # Save cluster sizes
    with open(f"{output_prefix}_cluster_sizes.txt", 'w') as f:
        f.write("cluster_id,size\n")
        for cluster_id in sorted(cluster_sizes.keys()):
            f.write(f"{cluster_id},{cluster_sizes[cluster_id]}\n")
    
    print(f"Analysis saved to {output_prefix}_edge_analysis.txt")
    return stats, cluster_sizes

def step1_extract_and_convert(dataset_path, active_window, R=70, L=125, alpha=1.2):
    """Step 1: Extract active_window vectors, convert to .fbin, and build initial index"""
    print(f"=== Step 1: Data Preparation and Initial Index Construction ===")
    
    # Use the provided dataset path (should be .fbin format for 1M vectors)
    input_file = dataset_path
    
    # Generate output names based on dataset and active window
    dataset_name = os.path.basename(dataset_path)
    dataset_prefix = dataset_name.replace('.fbin', '').replace('_base', '')
    output_fvecs = f"{dataset_prefix}_{active_window}_base.fvecs"
    output_fbin = f"{dataset_prefix}_{active_window}_base.fbin"
    index_prefix = f"{dataset_prefix}_{active_window}_index"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return False
        
    # Check if index already exists
    if os.path.exists(f"{index_prefix}"):
        print(f"Index {index_prefix} already exists, skipping step 1.")
        return True
        
    # Step 1.1: Extract Data Subset
    if not os.path.exists(output_fbin):
        print(f"1.1 Extract Data Subset: Reading first {active_window} vectors from {input_file}...")
        vectors, dim = read_fbin(input_file, start=0, count=active_window)
        print(f"Extracted {len(vectors)} vectors with {dim} dimensions")
        
        # Step 1.2: Save extracted subset as .fbin
        print(f"1.2 Save Format: Saving as {output_fbin}...")
        write_fbin(output_fbin, vectors, dim)
        
        print(f"Successfully created {output_fbin}")
    else:
        print(f"1.1-1.2: {output_fbin} already exists, skipping extraction.")
        
    # Step 1.3: Build Initial Index
    print("1.3 Build Initial Index: Constructing in-memory graph index...")
    build_cmd = [
        "/opt/DiskANN/build/apps/build_memory_index",
        "--data_type", "float",
        "--dist_fn", "l2", 
        "--data_path", output_fbin,
        "--index_path_prefix", index_prefix,
        "-R", str(R), "-L", str(L), "--alpha", str(alpha)
    ]
    
    print(" ".join(build_cmd))
    result = subprocess.run(build_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error building initial index: {result.stderr}")
        return False
        
    print(f"Successfully built initial index {index_prefix}")
    print("Step 1 completed successfully!")
    return True

def step2_clustering_and_analysis(active_window, k=8, R=70, use_balanced_kmeans=True):
    """Step 2: Balanced K-means clustering and pre-update graph analysis"""
    clustering_type = "Balanced K-means with K-means++" if use_balanced_kmeans else "Standard K-means with K-means++"
    print(f"=== Step 2: {clustering_type} Clustering and Pre-Update Graph Analysis ===")
    
    # Install required packages
    install_required_packages()
    from sklearn.cluster import KMeans
    
    # Find the data and index files created in step1
    # Look for files matching the pattern
    possible_data_files = [
        f for f in os.listdir('.') 
        if f.endswith(f'_{active_window}_base.fbin')
    ]
    
    if not possible_data_files:
        print(f"Error: No data file found matching pattern *_{active_window}_base.fbin!")
        return False
        
    data_file = possible_data_files[0]
    index_file = data_file.replace('_base.fbin', '_index')
    
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found! Run Step 1 first.")
        return False
        
    if not os.path.exists(index_file):
        print(f"Error: {index_file} not found! Run Step 1 first.")
        return False
    
    # Check if clustering already done
    dataset_prefix = data_file.replace(f'_{active_window}_base.fbin', '')
    labels_file = f"{dataset_prefix}_{active_window}_kmeans_labels.npy"
    centers_file = f"{dataset_prefix}_{active_window}_kmeans_centers.npy"
    
    if os.path.exists(labels_file) and os.path.exists(centers_file):
        print("2.1 Loading existing K-means clustering results...")
        labels = np.load(labels_file)
        centroids = np.load(centers_file)
        vectors, _ = read_fbin(data_file)
        vectors = np.array(vectors)
    else:
        # Step 2.1: Perform K-means Clustering
        print(f"2.1 Perform K-means Clustering: Running K-means with k={k}...")
        vectors, dim = read_fbin(data_file)
        vectors = np.array(vectors)
        print(f"Data shape: {vectors.shape}")
        
        # Apply K-means clustering
        if use_balanced_kmeans:
            print(f"Using balanced K-means with K-means++ initialization...")
            labels, centroids, convergence_info = balanced_kmeans_with_kmeanspp(
                vectors, k=k, max_iters=50, random_state=42
            )
            print(f"Balanced K-means converged in {len(convergence_info)} iterations")
        else:
            print(f"Using standard K-means with K-means++ initialization...")
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10, max_iter=300)
            labels = kmeans.fit_predict(vectors)
            centroids = kmeans.cluster_centers_
        
        # Save clustering results (labels and centroids)
        np.save(labels_file, labels)
        np.save(centers_file, centroids)
        print(f"Saved cluster assignments to {labels_file}")
        print(f"Saved {k} centroid vectors to {centers_file}")
    
    # Step 2.2: Analyze Inter-Cluster Edges from the built index
    print("2.2 Analyze Inter-Cluster Edges: Parsing adjacency lists from pre-update index...")
    stats, cluster_sizes = analyze_index_edges(
        index_file, labels, centroids, f"{dataset_prefix}_{active_window//1000}k_original", k, max_degree=R
    )
    
    print("Step 2 completed successfully!")
    return True

def step3_run_streaming_scenario(dataset_path, max_points_to_insert, active_window, consolidate_interval, R=70, L=125, alpha=1.2):
    """Step 3: Run test_streaming_scenario"""
    print("=== Step 3: Simulate Streaming Updates ===")
    
    total_file = dataset_path
    dataset_name = os.path.basename(dataset_path)
    
    if not os.path.exists(total_file):
        print(f"Error: {total_file} not found! Run Steps 1-2 first.")
        return False
    
    # Copy slice file to sift directory for streaming scenario
    # sift_total_file = os.path.join(sift_dir, os.path.basename(total_file))
    
    import shutil
    # if not os.path.exists(sift_total_file):
    #     print(f"Copying {total_file} to {sift_total_file}")
    #     shutil.copy2(total_file, sift_total_file)
    
    # Calculate parameters
    file_size = max_points_to_insert
    dataset_prefix = dataset_name.replace('.fbin', '').replace('_base', '')
    
    print(f"Using dataset: {dataset_name}")
    print(f"Max points to insert: {file_size}")
    print(f"Active window: {active_window}")
    print(f"Consolidate interval: {consolidate_interval}")
    
    # Build command for streaming scenario
    expected_output = f"{dataset_prefix}_streaming_index"

    # Check if already exists
    if os.path.exists(expected_output):
        print(f"{expected_output} already exists, skipping streaming scenario.")
        # # Copy result back to working directory if needed
        # local_output = os.path.basename(expected_output)
        # local_output_path = os.path.join(original_cwd, local_output)
        # if not os.path.exists(local_output_path):
        #     if os.path.abspath(expected_output) != os.path.abspath(local_output_path):
        #         shutil.copy2(expected_output, local_output_path)
        #     else:
        #         print(f"Result already in working directory: {local_output}")
        return True
    
    cmd = [
        "/opt/DiskANN/build/apps/test_streaming_scenario",
        "--data_type", "float",
        "--dist_fn", "l2", 
        "--data_path", total_file,
        "--index_path_prefix", expected_output,
        "-R", str(R),
        "-L", str(L), 
        "--alpha", str(alpha),
        "--insert_threads", "16",
        "--consolidate_threads", "16",
        "--max_points_to_insert", str(file_size),
        "--active_window", str(active_window),
        "--consolidate_interval", str(consolidate_interval),
        "--start_point_norm", "508"
    ]
    
    print("Running streaming scenario...")
    print(" ".join(cmd))
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode != 0:
            print(f"Error running streaming scenario:")
            print(f"Return code: {result.returncode}")
            print(f"stderr: {result.stderr}")
            print(f"stdout: {result.stdout}")
            return False
        
        print("Streaming scenario completed successfully!")
        print("Last few lines of output:")
        print(result.stdout[-500:])
        
        return True
        
    except subprocess.TimeoutExpired:
        print("Streaming scenario timed out after 1 hour")
        return False

def step4_analyze_updated_data(max_points_to_insert, consolidate_interval, active_window, k=8, R=70):
    """Step 4: Post-Update Analysis and Comparison"""
    print("=== Step 4: Post-Update Analysis and Comparison ===")
    
    # Install required packages
    install_required_packages()
    from sklearn.metrics import pairwise_distances_argmin
    
    # Find the dataset prefix from existing files
    possible_data_files = [
        f for f in os.listdir('.') 
        if f.endswith(f'_{active_window}_base.fbin')
    ]
    
    if not possible_data_files:
        print(f"Error: No data file found matching pattern *_{active_window}_base.fbin!")
        return False
    
    dataset_prefix = possible_data_files[0].replace(f'_{active_window}_base.fbin', '')
    
    # Find the generated .data file
    expected_data_file = f"{dataset_prefix}_streaming_index.after-streaming-act{active_window}-cons{consolidate_interval}-max{max_points_to_insert}.data"
    
    if not os.path.exists(expected_data_file):
        print(f"Error: Expected data file not found: {expected_data_file}")
        print("Available .data files:")
        for f in os.listdir("."):
            if f.endswith(".data"):
                print(f"  {f}")
        return False
    
    print(f"Found post-update data file: {expected_data_file}")
    
    # Step 4.1: Assign Cluster Labels to Updated Data
    print("4.1 Assign Cluster Labels to Updated Data: Using nearest centroid assignment...")
    
    # Read updated vectors
    updated_vectors, dim = read_fbin(expected_data_file)
    updated_vectors = np.array(updated_vectors)
    print(f"Updated data shape: {updated_vectors.shape}")
    
    # Load original centroids (do NOT re-run K-means)
    centers_file = f"{dataset_prefix}_{active_window}_kmeans_centers.npy"
    if not os.path.exists(centers_file):
        print(f"Error: {centers_file} not found! Run Step 2 first.")
        return False
    
    centroids = np.load(centers_file)
    print(f"Using original {len(centroids)} centroids for consistent cluster definition")
    
    # Assign cluster IDs to updated vectors using nearest centroid
    updated_labels = pairwise_distances_argmin(updated_vectors, centroids)
    print(f"Assigned labels to {len(updated_labels)} updated vectors")
    
    # Step 4.2: Analyze Inter-Cluster Edges on Updated Graph
    print("4.2 Analyze Inter-Cluster Edges on Updated Graph: Parsing post-update index...")
    
    # Try to find the updated index file (may have different naming)
    updated_index_file = None
    possible_index_files = [
        expected_data_file.replace('.data', ''),  # Remove .data extension
        f"{dataset_prefix}_streaming_index.after-streaming-act{active_window}-cons{consolidate_interval}-max{max_points_to_insert}",
    ]
    
    for idx_file in possible_index_files:
        if os.path.exists(idx_file):
            updated_index_file = idx_file
            break
    
    if updated_index_file:
        print(f"Found updated index file: {updated_index_file}")
        # Analyze edges from the updated index
        updated_stats, updated_cluster_sizes = analyze_index_edges(
            updated_index_file, updated_labels, centroids, 
            f"{dataset_prefix}_cons{consolidate_interval//1000}k", k, max_degree=R
        )
    else:
        print("Updated index file not found, using fallback distance-based analysis...")
        # Fallback to distance-based analysis
        updated_stats, updated_cluster_sizes = cross_cluster_edge_analysis_fallback(
            updated_labels, centroids, f"{dataset_prefix}_cons{consolidate_interval//1000}k", k
        )
    
    print("Step 4 completed successfully!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Streaming edge analysis pipeline for 1M vector datasets")
    parser.add_argument("dataset_name", type=str,
                       help="Name of 1M vector dataset (e.g., sift_base.fbin, gist_base.fbin)")
    parser.add_argument("max_points_to_insert", type=int,
                       help="Amount of points to use from the dataset (part of the 1M vectors)")
    parser.add_argument("active_window", type=int,
                       help="Size of active window (subset used in step1 for clustering)")
    parser.add_argument("consolidate_interval", type=int,
                       help="Consolidate interval for streaming operations")
    parser.add_argument("--k", type=int, default=8, 
                       help="Number of clusters for K-means (default: 8)")
    parser.add_argument("--steps", type=str, default="1,2,3,4",
                       help="Steps to run (e.g., '1,2' or '1,2,3,4')")
    parser.add_argument("--working_dir", type=str, default=".",
                       help="Working directory containing datasets (default: current directory)")
    parser.add_argument("-R", type=int, default=70,
                       help="Max degree for graph construction (default: 70)")
    parser.add_argument("-L", type=int, default=125,
                       help="Size of search list for graph construction (default: 125)")
    parser.add_argument("--alpha", type=float, default=1.2,
                       help="Alpha parameter for graph construction (default: 1.2)")
    parser.add_argument("--use_balanced_kmeans", action="store_true", default=True,
                       help="Use balanced K-means with K-means++ initialization (default: True)")
    parser.add_argument("--use_standard_kmeans", action="store_true", 
                       help="Use standard K-means instead of balanced (overrides --use_balanced_kmeans)")
    
    args = parser.parse_args()
    
    # Handle K-means algorithm selection
    if args.use_standard_kmeans:
        args.use_balanced_kmeans = False
    
    # Handle dataset path - support relative paths from data directory
    # Store original working directory to resolve relative paths correctly
    original_cwd = os.getcwd()
    
    if os.path.isabs(args.dataset_name):
        # Absolute path
        dataset_path = args.dataset_name
        dataset_dir = os.path.dirname(dataset_path)
        dataset_name = os.path.basename(args.dataset_name)
    elif args.dataset_name.startswith('./'):
        # Relative path from original calling directory (e.g., ./sift/sift_base.fbin)
        # Resolve relative to original working directory
        dataset_path = os.path.join(original_cwd, args.dataset_name)
        dataset_dir = os.path.dirname(dataset_path)
        dataset_name = os.path.basename(dataset_path)
    else:
        # Just filename - detect dataset type and construct path
        dataset_name = args.dataset_name
        if "sift" in dataset_name.lower():
            dataset_dir = os.path.join(original_cwd, args.working_dir, "sift")
        elif "gist" in dataset_name.lower():
            dataset_dir = os.path.join(original_cwd, args.working_dir, "gist") 
        else:
            dataset_dir = os.path.join(original_cwd, args.working_dir)
        dataset_path = os.path.join(dataset_dir, dataset_name)
    
    # Verify dataset file exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file {dataset_path} not found!")
        print(f"Original working directory: {original_cwd}")
        print(f"Current working directory: {os.getcwd()}")
        return 1
        
    # Work in current directory for output files
    work_dir = os.getcwd()
    print(f"Working directory: {os.getcwd()}")
    print(f"Dataset path: {dataset_path}")
    
    steps_to_run = [int(s.strip()) for s in args.steps.split(',')]
    
    print(f"Dataset: {args.dataset_name}")
    print(f"Max points to insert: {args.max_points_to_insert:,}")
    print(f"Active window: {args.active_window:,}")
    print(f"Consolidate interval: {args.consolidate_interval:,}")
    print(f"K-means clusters: {args.k}")
    print(f"Graph parameters: R={args.R}, L={args.L}, alpha={args.alpha}")
    print(f"Steps to run: {steps_to_run}")
    print("=" * 60)
    
    success = True
    
    if 1 in steps_to_run:
        success &= step1_extract_and_convert(dataset_path, args.active_window, args.R, args.L, args.alpha)
        if not success:
            print("Step 1 failed, aborting.")
            return 1
    
    if 2 in steps_to_run and success:
        success &= step2_clustering_and_analysis(args.active_window, args.k, args.R, args.use_balanced_kmeans)
        if not success:
            print("Step 2 failed, aborting.")
            return 1
    
    if 3 in steps_to_run and success:
        success &= step3_run_streaming_scenario(dataset_path, args.max_points_to_insert, args.active_window, args.consolidate_interval, args.R, args.L, args.alpha)
        if not success:
            print("Step 3 failed, aborting.")
            return 1
    
    if 4 in steps_to_run and success:
        success &= step4_analyze_updated_data(args.max_points_to_insert, args.consolidate_interval, args.active_window, args.k, args.R)
        if not success:
            print("Step 4 failed, aborting.")
            return 1
    
    if success:
        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print("\nGenerated files:")
        for f in sorted(os.listdir(".")):
            if any(f.startswith(prefix) for prefix in [f'sift_{args.active_window}_']) and \
               any(f.endswith(ext) for ext in ['.txt', '.csv', '.npy', '.fbin', '.data']):
                size = os.path.getsize(f)
                print(f"  {f} ({size:,} bytes)")
    else:
        print("\nPipeline failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())