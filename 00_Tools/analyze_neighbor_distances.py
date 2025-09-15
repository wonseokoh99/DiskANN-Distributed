import numpy as np
import struct
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

def get_graph_offsets(graph_file_path):
    """그래프 파일을 미리 스캔하여 각 노드의 이웃 리스트 시작 위치와 전체 노드 수를 반환합니다."""
    print("그래프 파일을 사전 스캔하여 노드별 오프셋을 생성합니다...")
    offsets = []
    with open(graph_file_path, 'rb') as f:
        # DiskANN _disk.graph 헤더 (num_points, max_degree) -> 이 헤더의 num_points는 신뢰하지 않음
        header_size = 16 
        f.seek(header_size)
        current_offset = header_size
        
        while True:
            # 현재 위치를 오프셋으로 추가
            offsets.append(current_offset)
            
            # 이웃 수 읽기
            num_neighbors_bytes = f.read(4)
            if not num_neighbors_bytes:
                offsets.pop() # 파일 끝에 도달했으면 마지막 오프셋은 유효하지 않음
                break
            
            num_neighbors = struct.unpack('<I', num_neighbors_bytes)[0]
            bytes_to_skip = num_neighbors * 4
            current_offset += (4 + bytes_to_skip)
            
            # 다음 이웃 리스트로 건너뛰기
            f.seek(bytes_to_skip, os.SEEK_CUR)

            # 파일의 끝인지 확인
            if not f.read(1):
                break
            f.seek(-1, os.SEEK_CUR) # 1바이트 다시 뒤로
                
    return np.array(offsets, dtype=np.int64)

def get_neighbors(graph_file_handle, node_id, offsets):
    """오프셋을 사용해 특정 노드의 이웃 ID 리스트를 반환합니다."""
    graph_file_handle.seek(offsets[node_id])
    num_neighbors = struct.unpack('<I', graph_file_handle.read(4))[0]
    if num_neighbors == 0:
        return np.array([], dtype=np.uint32)
    neighbors = np.fromfile(graph_file_handle, dtype=np.uint32, count=num_neighbors)
    return neighbors

# --- [수정] get_vector 함수가 data_start_offset을 사용하도록 변경 ---
def get_vector(data_file_handle, node_id, dim, data_start_offset, data_type=np.float32):
    """데이터 파일에서 시작 오프셋과 ID를 기반으로 벡터를 반환합니다."""
    bytes_per_vector = dim * np.dtype(data_type).itemsize
    # 계산된 최종 오프셋으로 이동
    offset = data_start_offset + node_id * bytes_per_vector
    data_file_handle.seek(offset)
    vector = np.fromfile(data_file_handle, dtype=data_type, count=dim)
    # 파일에서 충분한 데이터를 읽었는지 확인
    if vector.shape[0] < dim:
        return None
    return vector

def plot_distance_distribution(distances, output_path):
    """거리 분포를 시각화하고 통계 정보를 포함하여 저장합니다."""
    if not distances:
        print("경고: 분석할 거리 데이터가 없습니다.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 8))
    
    sns.histplot(distances, kde=True, bins=50, stat="density", color="skyblue")
    
    mean_dist = np.mean(distances)
    median_dist = np.median(distances)
    std_dist = np.std(distances)
    
    plt.axvline(mean_dist, color='red', linestyle='--', label=f'Mean: {mean_dist:.2f}')
    plt.axvline(median_dist, color='green', linestyle='-', label=f'Median: {median_dist:.2f}')
    
    plt.title('Distribution of Distances to Neighbors', fontsize=16)
    plt.xlabel('Euclidean Distance', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    
    stats_text = (f"Std Dev: {std_dist:.2f}\n"
                  f"Min: {np.min(distances):.2f}\n"
                  f"Max: {np.max(distances):.2f}\n"
                  f"Count: {len(distances)}")
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
             
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"거리 분포 그래프가 '{output_path}'에 저장되었습니다.")

def analyze_neighbor_distances(graph_file, data_file, output_prefix, dim, data_start_offset, sample_size=1000):
    """메인 분석 함수"""
    
    graph_offsets = get_graph_offsets(graph_file)
    num_nodes_in_graph = len(graph_offsets)
    print(f"그래프 파일에서 {num_nodes_in_graph}개의 노드를 확인했습니다.")
    
    # 분석할 노드 샘플링
    if sample_size > num_nodes_in_graph:
        sample_size = num_nodes_in_graph
    
    node_indices_to_analyze = np.random.choice(num_nodes_in_graph, size=sample_size, replace=False)
    
    all_distances = []
    
    print(f"{sample_size}개의 샘플 노드에 대한 이웃 간 거리 분석을 시작합니다...")
    with open(graph_file, 'rb') as gf, open(data_file, 'rb') as df:
        for node_id in tqdm(node_indices_to_analyze, desc="Analyzing distances"):
            try:
                source_vector = get_vector(df, node_id, dim, data_start_offset)
                if source_vector is None:
                    print(f"경고: 소스 노드 {node_id}의 벡터를 읽을 수 없습니다. 건너뜁니다.")
                    continue

                neighbor_ids = get_neighbors(gf, node_id, graph_offsets)
                
                for neighbor_id in neighbor_ids:
                    neighbor_vector = get_vector(df, neighbor_id, dim, data_start_offset)
                    if neighbor_vector is None:
                        # print(f"경고: 이웃 노드 {neighbor_id}의 벡터를 읽을 수 없습니다. 건너뜁니다.")
                        continue
                    
                    distance = np.linalg.norm(source_vector - neighbor_vector)
                    all_distances.append(distance)
            except Exception as e:
                print(f"오류: 노드 {node_id} 처리 중 문제 발생 - {e}")

    # 시각화 및 데이터 저장
    plot_output_path = f"{output_prefix}_distance_distribution.png"
    plot_distance_distribution(all_distances, plot_output_path)
    
    distance_csv_path = f"{output_prefix}_distances.csv"
    pd.DataFrame(all_distances, columns=['distance']).to_csv(distance_csv_path, index=False)
    print(f"계산된 거리 데이터가 '{distance_csv_path}'에 저장되었습니다.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DiskANN 그래프의 이웃 간 거리 분포를 분석하고 시각화합니다.")
    parser.add_argument('--graph_file', type=str, required=True, help='입력 그래프 파일 경로 (e.g., _disk.graph)')
    parser.add_argument('--data_file', type=str, required=True, help='원본 벡터 데이터 파일 경로 (.bin, .fbin 등)')
    parser.add_argument('--output_prefix', type=str, required=True, help='분석 결과를 저장할 파일들의 접두사')
    parser.add_argument('--dim', type=int, required=True, help='벡터의 차원')
    # --- [수정] 오프셋 인자 추가, 기본값 8 (fbin 헤더) ---
    parser.add_argument('--data_start_offset', type=int, default=8, help='데이터 파일 내 벡터 데이터의 시작 바이트 오프셋')
    parser.add_argument('--sample_size', type=int, default=1000, help='분석할 노드의 샘플 크기 (기본값: 1000)')
    
    args = parser.parse_args()

    analyze_neighbor_distances(
        graph_file=args.graph_file,
        data_file=args.data_file,
        output_prefix=args.output_prefix,
        dim=args.dim,
        data_start_offset=args.data_start_offset,
        sample_size=args.sample_size
    )