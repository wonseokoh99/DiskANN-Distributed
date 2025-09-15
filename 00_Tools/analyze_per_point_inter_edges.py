import numpy as np
import pandas as pd
import struct
import argparse
import os
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def load_partitions(partition_file_path, size_t_bytes=8):
    """파티션 파일을 로드하여 NumPy 배열로 반환합니다."""
    print(f"'{partition_file_path}'에서 파티션 정보를 로딩합니다...")
    metadata_format = 'Q' if size_t_bytes == 8 else 'I'
    with open(partition_file_path, 'rb') as f:
        metadata_size = size_t_bytes * 2
        num_points, _ = struct.unpack(f'<{metadata_format}{metadata_format}', f.read(metadata_size))
        partition_labels = np.fromfile(f, dtype=np.int32, count=num_points)
    print(f"성공: {len(partition_labels)}개 노드에 대한 파티션 정보를 로드했습니다.")
    return partition_labels

def get_graph_offsets(graph_file_path):
    """그래프 파일을 미리 스캔하여 각 노드의 이웃 리스트 시작 위치를 반환합니다."""
    print("그래프 파일을 사전 스캔하여 노드별 오프셋을 생성합니다...")
    offsets = []
    with open(graph_file_path, 'rb') as f:
        header_size = 16 
        f.seek(header_size)
        current_offset = header_size
        while True:
            offsets.append(current_offset)
            num_neighbors_bytes = f.read(4)
            if not num_neighbors_bytes:
                offsets.pop()
                break
            num_neighbors = struct.unpack('<I', num_neighbors_bytes)[0]
            bytes_to_skip = num_neighbors * 4
            current_offset += (4 + bytes_to_skip)
            f.seek(bytes_to_skip, os.SEEK_CUR)
            if not f.read(1): break
            f.seek(-1, os.SEEK_CUR)
    return np.array(offsets, dtype=np.int64)

def get_neighbors(graph_file_handle, node_id, offsets):
    """오프셋을 사용해 특정 노드의 이웃 ID 리스트를 반환합니다."""
    if node_id >= len(offsets):
        return np.array([], dtype=np.uint32)
    graph_file_handle.seek(offsets[node_id])
    num_neighbors = struct.unpack('<I', graph_file_handle.read(4))[0]
    if num_neighbors == 0:
        return np.array([], dtype=np.uint32)
    return np.fromfile(graph_file_handle, dtype=np.uint32, count=num_neighbors)

# --- [수정] y_max 인자 추가 ---
def create_inter_edge_distribution_plots(df, output_prefix, y_max=None):
    """
    파티션 쌍별 inter-edge 개수 분포를 격자 히스토그램으로 시각화합니다.
    """
    print("Inter-edge 개수 분포 시각화를 생성합니다...")
    
    df_cleaned = df.dropna(subset=['source_partition', 'target_partition'])

    # --- [수정] sharey=True로 변경하여 Y축을 공유하도록 설정 ---
    g = sns.FacetGrid(df_cleaned, row="source_partition", col="target_partition", 
                      margin_titles=True, sharex=False, sharey=True)
                      
    g.map(plt.hist, "edge_count", bins=10, edgecolor="w")
    g.set_axis_labels("Edge Count per Node", "Number of Nodes")
    g.set_titles(col_template="Target: {col_name}", row_template="Source: {row_name}")
    g.fig.suptitle('Distribution of Inter-Edge Counts per Node (Source -> Target)', y=1.02)
    
    # --- [수정] y_max가 지정된 경우 Y축 최대값 고정 ---
    if y_max is not None:
        g.set(ylim=(0, y_max))
    
    plot_path = f"{output_prefix}_inter_edge_distribution_grid.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" -> 분포 시각화가 '{plot_path}'에 저장되었습니다.")

# --- [수정] y_max 인자 추가 ---
def analyze_per_point_inter_edges(graph_file, partition_file, output_prefix, size_t_bytes, y_max=None):
    """
    각 포인트에서 다른 파티션으로 나가는 inter-edge를 분석하고 통계를 생성합니다.
    """
    partition_labels = load_partitions(partition_file, size_t_bytes)
    graph_offsets = get_graph_offsets(graph_file)
    num_nodes = len(partition_labels)

    results = []

    with open(graph_file, 'rb') as gf:
        for source_node_id in tqdm(range(num_nodes), desc="Analyzing per-point inter-edges"):
            if source_node_id >= len(graph_offsets): continue

            raw_source_label = partition_labels[source_node_id]
            source_partition = raw_source_label if raw_source_label >= 0 else -raw_source_label - 1
            is_source_deleted = raw_source_label < 0

            neighbors = get_neighbors(gf, source_node_id, graph_offsets)
            if len(neighbors) == 0: continue

            target_partition_counts = defaultdict(int)

            for target_node_id in neighbors:
                if target_node_id >= len(partition_labels): continue
                
                raw_target_label = partition_labels[target_node_id]
                target_partition = raw_target_label if raw_target_label >= 0 else -raw_target_label - 1

                if source_partition != target_partition:
                    target_partition_counts[target_partition] += 1
            
            if target_partition_counts:
                for target_p, count in target_partition_counts.items():
                    results.append({
                        'source_node_id': source_node_id,
                        'source_partition': source_partition,
                        'is_source_deleted': is_source_deleted,
                        'target_partition': target_p,
                        'edge_count': count
                    })

    if not results:
        print("분석 결과: Inter-partition 엣지를 찾을 수 없습니다.")
        return

    df = pd.DataFrame(results)
    full_stats_path = f"{output_prefix}_full_stats.csv"
    df.to_csv(full_stats_path, index=False)
    print(f"\n전체 상세 분석 결과가 '{full_stats_path}'에 저장되었습니다.")

    summary_path = f"{output_prefix}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=== Per-Point Inter-Edge Analysis Summary ===\n\n")
        top_connections = df.sort_values(by='edge_count', ascending=False).head(20)
        f.write("--- Top 20 Single Point Connections to a Target Partition ---\n")
        f.write(top_connections.to_string(index=False))
        f.write("\n\n")
        f.write("--- Top Connector Node for Each Partition Pair (Source -> Target) ---\n")
        top_connectors = df.loc[df.groupby(['source_partition', 'target_partition'])['edge_count'].idxmax()]
        top_connectors = top_connectors.sort_values(by=['source_partition', 'target_partition'])
        f.write(top_connectors.to_string(index=False))
    print(f"요약 통계가 '{summary_path}'에 저장되었습니다.")

    # --- [수정] y_max 인자 전달 ---
    create_inter_edge_distribution_plots(df, output_prefix, y_max)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="각 포인트의 inter-partition edge를 상세 분석합니다.")
    parser.add_argument('--graph_file', type=str, required=True, help='입력 그래프 파일 경로 (e.g., _disk.graph)')
    parser.add_argument('--partition_file', type=str, required=True, help='파티션 정보 파일 경로 (.partition)')
    parser.add_argument('--output_prefix', type=str, required=True, help='분석 결과를 저장할 파일들의 접두사')
    parser.add_argument('--size_t_bytes', type=int, default=8, choices=[4, 8], help='파일을 생성한 C++ 환경의 size_t 타입 크기')
    # --- [수정] y_max 인자 추가 ---
    parser.add_argument('--y_max', type=int, default=None, help='분포도 Y축의 최대값을 고정합니다 (기본값: 자동 조정)')
    
    args = parser.parse_args()

    analyze_per_point_inter_edges(
        graph_file=args.graph_file,
        partition_file=args.partition_file,
        output_prefix=args.output_prefix,
        size_t_bytes=args.size_t_bytes,
        y_max=args.y_max
    )