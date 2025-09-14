#!/usr/bin/env python3
"""
Graph Connectivity Comparison Tool

두 개의 DiskANN 그래프 CSV 파일을 비교하여 연결성 차이를 분석합니다.
같은 dataset_offset을 가진 노드들 간의 이웃 차이를 분석합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from collections import defaultdict, Counter
import json

def load_graph_csv(csv_file):
    """CSV 파일에서 그래프 데이터를 로드"""
    print(f"Loading graph from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # neighbors 컬럼을 파싱 (빈 문자열 처리)
    graph_dict = {}
    for _, row in df.iterrows():
        dataset_offset = row['dataset_offset']
        neighbors_str = row['neighbors']
        
        if pd.isna(neighbors_str) or neighbors_str == '':
            neighbors = []
        else:
            neighbors = [int(x) for x in str(neighbors_str).split(',')]
        
        graph_dict[dataset_offset] = {
            'node_id': row['node_id'],
            'degree': row['degree'],
            'neighbors': set(neighbors)  # set으로 변환하여 교집합/차집합 연산 용이
        }
    
    print(f"  Loaded {len(graph_dict)} nodes")
    return graph_dict

def analyze_neighbor_differences(graph1, graph2):
    """두 그래프 간의 이웃 차이 분석"""
    print("Analyzing neighbor differences...")
    
    # 공통 노드 찾기
    common_offsets = set(graph1.keys()) & set(graph2.keys())
    print(f"Common nodes: {len(common_offsets)}")
    
    if len(common_offsets) == 0:
        print("No common nodes found!")
        return None
    
    analysis_results = []
    
    for offset in common_offsets:
        node1 = graph1[offset]
        node2 = graph2[offset]
        
        neighbors1 = node1['neighbors']
        neighbors2 = node2['neighbors']
        
        # 교집합, 차집합 계산
        intersection = neighbors1 & neighbors2
        only_in_graph1 = neighbors1 - neighbors2
        only_in_graph2 = neighbors2 - neighbors1
        union = neighbors1 | neighbors2
        
        # Jaccard 유사도 계산
        jaccard_similarity = len(intersection) / len(union) if len(union) > 0 else 0
        
        # 결과 저장
        result = {
            'dataset_offset': offset,
            'degree_graph1': node1['degree'],
            'degree_graph2': node2['degree'],
            'degree_diff': node2['degree'] - node1['degree'],
            'common_neighbors': len(intersection),
            'only_in_graph1': len(only_in_graph1),
            'only_in_graph2': len(only_in_graph2),
            'total_unique_neighbors': len(union),
            'jaccard_similarity': jaccard_similarity,
            'neighbor_overlap_ratio': len(intersection) / max(len(neighbors1), len(neighbors2)) if max(len(neighbors1), len(neighbors2)) > 0 else 0
        }
        
        analysis_results.append(result)
    
    return pd.DataFrame(analysis_results)

def generate_summary_statistics(df):
    """요약 통계 생성"""
    print("\n=== SUMMARY STATISTICS ===")
    
    # 기본 통계
    print("Degree Comparison:")
    print(f"  Average degree in Graph 1: {df['degree_graph1'].mean():.2f}")
    print(f"  Average degree in Graph 2: {df['degree_graph2'].mean():.2f}")
    print(f"  Average degree difference: {df['degree_diff'].mean():.2f}")
    
    print("\nNeighbor Overlap:")
    print(f"  Average Jaccard similarity: {df['jaccard_similarity'].mean():.4f}")
    print(f"  Average neighbor overlap ratio: {df['neighbor_overlap_ratio'].mean():.4f}")
    print(f"  Average common neighbors: {df['common_neighbors'].mean():.2f}")
    
    print("\nUnique Neighbors:")
    print(f"  Average neighbors only in Graph 1: {df['only_in_graph1'].mean():.2f}")
    print(f"  Average neighbors only in Graph 2: {df['only_in_graph2'].mean():.2f}")
    
    # 분포 통계
    print("\nDistribution Analysis:")
    print(f"  Nodes with identical neighbors: {(df['jaccard_similarity'] == 1.0).sum()}")
    print(f"  Nodes with no common neighbors: {(df['jaccard_similarity'] == 0.0).sum()}")
    print(f"  Nodes with degree increase: {(df['degree_diff'] > 0).sum()}")
    print(f"  Nodes with degree decrease: {(df['degree_diff'] < 0).sum()}")
    
    return {
        'avg_degree_graph1': float(df['degree_graph1'].mean()),
        'avg_degree_graph2': float(df['degree_graph2'].mean()),
        'avg_jaccard_similarity': float(df['jaccard_similarity'].mean()),
        'avg_neighbor_overlap_ratio': float(df['neighbor_overlap_ratio'].mean()),
        'avg_common_neighbors': float(df['common_neighbors'].mean()),
        'avg_only_in_graph1': float(df['only_in_graph1'].mean()),
        'avg_only_in_graph2': float(df['only_in_graph2'].mean()),
        'identical_neighbors_count': int((df['jaccard_similarity'] == 1.0).sum()),
        'no_common_neighbors_count': int((df['jaccard_similarity'] == 0.0).sum()),
        'degree_increase_count': int((df['degree_diff'] > 0).sum()),
        'degree_decrease_count': int((df['degree_diff'] < 0).sum()),
        'total_nodes_analyzed': int(len(df))
    }

def create_visualizations(df, output_prefix):
    """시각화 생성"""
    print("Creating visualizations...")
    
    # 스타일 설정
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Degree 비교 산점도
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Degree comparison scatter plot
    axes[0, 0].scatter(df['degree_graph1'], df['degree_graph2'], alpha=0.6)
    axes[0, 0].plot([0, max(df['degree_graph1'].max(), df['degree_graph2'].max())], 
                    [0, max(df['degree_graph1'].max(), df['degree_graph2'].max())], 
                    'r--', alpha=0.8)
    axes[0, 0].set_xlabel('Degree in Graph 1')
    axes[0, 0].set_ylabel('Degree in Graph 2')
    axes[0, 0].set_title('Degree Comparison')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Degree difference histogram
    axes[0, 1].hist(df['degree_diff'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.8)
    axes[0, 1].set_xlabel('Degree Difference (Graph2 - Graph1)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Degree Difference Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Jaccard similarity histogram
    axes[1, 0].hist(df['jaccard_similarity'], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Jaccard Similarity')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Neighbor Similarity Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Neighbor overlap ratio vs degree difference
    scatter = axes[1, 1].scatter(df['degree_diff'], df['jaccard_similarity'], 
                                c=df['degree_graph1'], alpha=0.6, cmap='viridis')
    axes[1, 1].set_xlabel('Degree Difference')
    axes[1, 1].set_ylabel('Jaccard Similarity')
    axes[1, 1].set_title('Similarity vs Degree Change')
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 1], label='Original Degree (Graph 1)')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_comparison_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 상세 분석 차트
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Common neighbors vs total neighbors
    axes[0, 0].scatter(df['total_unique_neighbors'], df['common_neighbors'], alpha=0.6)
    axes[0, 0].set_xlabel('Total Unique Neighbors')
    axes[0, 0].set_ylabel('Common Neighbors')
    axes[0, 0].set_title('Common vs Total Neighbors')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Unique neighbors comparison
    axes[0, 1].scatter(df['only_in_graph1'], df['only_in_graph2'], alpha=0.6)
    max_unique = max(df['only_in_graph1'].max(), df['only_in_graph2'].max())
    axes[0, 1].plot([0, max_unique], [0, max_unique], 'r--', alpha=0.8)
    axes[0, 1].set_xlabel('Unique Neighbors in Graph 1')
    axes[0, 1].set_ylabel('Unique Neighbors in Graph 2')
    axes[0, 1].set_title('Unique Neighbors Comparison')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Box plots for different similarity ranges
    similarity_ranges = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
    degree_diff_by_similarity = []
    
    for i, range_label in enumerate(similarity_ranges):
        lower = i * 0.2
        upper = (i + 1) * 0.2
        mask = (df['jaccard_similarity'] >= lower) & (df['jaccard_similarity'] < upper)
        if i == 4:  # Last range includes 1.0
            mask = (df['jaccard_similarity'] >= lower) & (df['jaccard_similarity'] <= upper)
        degree_diff_by_similarity.append(df[mask]['degree_diff'].values)
    
    axes[1, 0].boxplot(degree_diff_by_similarity, labels=similarity_ranges)
    axes[1, 0].set_xlabel('Jaccard Similarity Range')
    axes[1, 0].set_ylabel('Degree Difference')
    axes[1, 0].set_title('Degree Change by Similarity Level')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Neighbor overlap ratio distribution
    axes[1, 1].hist(df['neighbor_overlap_ratio'], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Neighbor Overlap Ratio')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Neighbor Overlap Ratio Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved as {output_prefix}_comparison_overview.png and {output_prefix}_detailed_analysis.png")

def main():
    parser = argparse.ArgumentParser(description='Compare connectivity between two DiskANN graphs')
    parser.add_argument('--graph1_csv', required=True, help='First graph CSV file')
    parser.add_argument('--graph2_csv', required=True, help='Second graph CSV file')
    parser.add_argument('--output_prefix', required=True, help='Output file prefix')
    parser.add_argument('--save_detailed_csv', action='store_true', help='Save detailed comparison CSV')
    
    args = parser.parse_args()
    
    # 그래프 로드
    graph1 = load_graph_csv(args.graph1_csv)
    graph2 = load_graph_csv(args.graph2_csv)
    
    # 이웃 차이 분석
    comparison_df = analyze_neighbor_differences(graph1, graph2)
    
    if comparison_df is None:
        print("Analysis failed due to no common nodes.")
        return
    
    # 요약 통계 생성
    summary_stats = generate_summary_statistics(comparison_df)
    
    # 요약 통계 저장
    with open(f'{args.output_prefix}_summary.json', 'w') as f:
        json.dump(summary_stats, f, indent=2)
    print(f"Summary statistics saved to {args.output_prefix}_summary.json")
    
    # 상세 CSV 저장 (옵션)
    if args.save_detailed_csv:
        comparison_df.to_csv(f'{args.output_prefix}_detailed_comparison.csv', index=False)
        print(f"Detailed comparison saved to {args.output_prefix}_detailed_comparison.csv")
    
    # 시각화 생성
    create_visualizations(comparison_df, args.output_prefix)
    
    print(f"\nAnalysis complete! Check {args.output_prefix}_* files for results.")

if __name__ == "__main__":
    main()