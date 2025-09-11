import numpy as np
import pandas as pd
import struct
import argparse
import os
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 각 워커 프로세스가 실행할 함수
def process_chunk(args_tuple):
    """
    할당된 노드 청크와 오프셋을 사용해 파일에서 직접 데이터를 읽고 이중 집계 로직으로 처리합니다.
    """
    node_ids_chunk, offsets, graph_file_path, partition_labels, num_partitions = args_tuple
    
    matrix_size = num_partitions + 1
    deleted_partition_idx = num_partitions 
    local_matrix = np.zeros((matrix_size, matrix_size), dtype=np.int64)
    
    # Out-degree 정보를 추가로 수집
    local_out_degrees = {}

    with open(graph_file_path, 'rb') as f:
        for source_node_id in node_ids_chunk:
            try:
                f.seek(offsets[source_node_id])
                num_neighbors_bytes = f.read(4)
                if not num_neighbors_bytes or len(num_neighbors_bytes) < 4: continue
                
                num_neighbors = struct.unpack('<I', num_neighbors_bytes)[0]
                
                # Out-degree 정보 저장
                local_out_degrees[source_node_id] = num_neighbors
                
                if num_neighbors > 0:
                    neighbors = np.fromfile(f, dtype=np.uint32, count=num_neighbors)
                    
                    raw_source_label = partition_labels[source_node_id]

                    for target_node_id in neighbors:
                        if target_node_id < len(partition_labels):
                            raw_target_label = partition_labels[target_node_id]

                            # --- [핵심 수정] 이중 집계 로직 ---
                            
                            # 1. 원본 파티션 ID 계산 (음수 -> 양수 변환)
                            source_orig_p = raw_source_label if raw_source_label >= 0 else -raw_source_label - 1
                            target_orig_p = raw_target_label if raw_target_label >= 0 else -raw_target_label - 1
                            
                            # 2. '원본 파티션' 기준으로 항상 집계
                            local_matrix[source_orig_p, target_orig_p] += 1
                            
                            # 3. 노드가 삭제된 경우, 'Deleted' 카테고리에 '추가' 집계
                            is_source_deleted = raw_source_label < 0
                            is_target_deleted = raw_target_label < 0
                            
                            if is_source_deleted and not is_target_deleted:
                                local_matrix[deleted_partition_idx, target_orig_p] += 1
                            elif not is_source_deleted and is_target_deleted:
                                local_matrix[source_orig_p, deleted_partition_idx] += 1
                            elif is_source_deleted and is_target_deleted:
                                local_matrix[deleted_partition_idx, deleted_partition_idx] += 1

            except IndexError:
                continue
            
    return local_matrix, local_out_degrees

def analyze_out_degree_outliers(out_degrees, partition_labels, output_prefix, percentile_threshold=95):
    """
    Out-degree outlier 분석 및 시각화
    """
    print(f"\nOut-degree outlier 분석을 시작합니다 (상위 {100-percentile_threshold}% 기준)...")
    
    # Out-degree 데이터 준비
    node_ids = list(out_degrees.keys())
    degrees = list(out_degrees.values())
    
    if not degrees:
        print("경고: Out-degree 데이터가 없습니다.")
        return
    
    # 통계 계산
    degree_stats = {
        'mean': np.mean(degrees),
        'median': np.median(degrees),
        'std': np.std(degrees),
        'min': np.min(degrees),
        'max': np.max(degrees),
        'q25': np.percentile(degrees, 25),
        'q75': np.percentile(degrees, 75),
        'q95': np.percentile(degrees, 95),
        'q99': np.percentile(degrees, 99)
    }
    
    # Outlier 임계값 결정
    threshold = np.percentile(degrees, percentile_threshold)
    
    # Outlier 노드들 식별
    outlier_data = []
    for node_id, degree in out_degrees.items():
        if degree >= threshold:
            raw_partition_label = partition_labels[node_id] if node_id < len(partition_labels) else -1
            original_partition = raw_partition_label if raw_partition_label >= 0 else -raw_partition_label - 1
            is_deleted = raw_partition_label < 0
            
            outlier_data.append({
                'node_id': node_id,
                'out_degree': degree,
                'raw_partition_label': raw_partition_label,
                'original_partition': original_partition,
                'is_deleted': is_deleted
            })
    
    outlier_df = pd.DataFrame(outlier_data)
    
    # 결과 출력
    print(f"\n=== Out-degree 통계 정보 ===")
    for key, value in degree_stats.items():
        print(f"{key.upper()}: {value:.2f}")
    
    print(f"\nOutlier 임계값 ({percentile_threshold}th percentile): {threshold:.2f}")
    print(f"총 outlier 노드 수: {len(outlier_df)}")
    
    if len(outlier_df) > 0:
        print(f"Outlier 중 삭제된 노드 수: {outlier_df['is_deleted'].sum()}")
        
        # 파티션별 outlier 분포
        partition_counts = outlier_df['original_partition'].value_counts().sort_index()
        print(f"\n=== 파티션별 Outlier 분포 ===")
        for partition, count in partition_counts.items():
            print(f"Partition {partition}: {count}개")
    
    # CSV 파일 저장
    outlier_csv_path = f"{output_prefix}_outliers.csv"
    with open(outlier_csv_path, 'w') as f:
        f.write("# Out-degree Statistics\n")
        for key, value in degree_stats.items():
            f.write(f"# {key}: {value:.4f}\n")
        f.write(f"# Outlier threshold ({percentile_threshold}th percentile): {threshold:.4f}\n")
        f.write(f"# Total outliers: {len(outlier_df)}\n\n")
        
        if len(outlier_df) > 0:
            outlier_df.to_csv(f, index=False)
    
    print(f"Outlier 분석 결과가 '{outlier_csv_path}'에 저장되었습니다.")
    
    # 시각화
    create_outlier_visualizations(degrees, outlier_df, threshold, degree_stats, output_prefix, percentile_threshold)
    
    return outlier_df, degree_stats

def create_outlier_visualizations(degrees, outlier_df, threshold, degree_stats, output_prefix, percentile_threshold):
    """
    Out-degree outlier 시각화
    """
    print("Out-degree 분포 시각화를 생성합니다...")
    
    # 1. Out-degree 히스토그램
    plt.figure(figsize=(12, 8))
    plt.hist(degrees, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                label=f'{percentile_threshold}th percentile: {threshold:.1f}')
    plt.axvline(degree_stats['mean'], color='green', linestyle='-', linewidth=2, 
                label=f'Mean: {degree_stats["mean"]:.1f}')
    plt.axvline(degree_stats['median'], color='orange', linestyle='-', linewidth=2, 
                label=f'Median: {degree_stats["median"]:.1f}')
    
    plt.xlabel('Out-degree')
    plt.ylabel('Frequency')
    plt.title('Out-degree Distribution with Outlier Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_prefix}_out_degree_histogram.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Log-scale 히스토그램 (높은 차이를 보기 위해)
    plt.figure(figsize=(12, 8))
    plt.hist(degrees, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                label=f'{percentile_threshold}th percentile: {threshold:.1f}')
    plt.yscale('log')
    plt.xlabel('Out-degree')
    plt.ylabel('Frequency (log scale)')
    plt.title('Out-degree Distribution (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_prefix}_out_degree_histogram_log.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Outlier 노드들의 파티션별 분포 (있는 경우에만)
    if len(outlier_df) > 0:
        plt.figure(figsize=(12, 8))
        partition_counts = outlier_df['original_partition'].value_counts().sort_index()
        
        bars = plt.bar(partition_counts.index, partition_counts.values, 
                      alpha=0.7, color='purple', edgecolor='black')
        plt.xlabel('Original Partition ID')
        plt.ylabel('Number of Outlier Nodes')
        plt.title(f'Outlier Nodes Distribution by Partition (>{percentile_threshold}th percentile)')
        plt.grid(True, alpha=0.3)
        
        # 값 표시
        for bar, count in zip(bars, partition_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom')
        
        plt.savefig(f"{output_prefix}_outliers_by_partition.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Top outlier들의 상세 정보 (상위 20개)
        if len(outlier_df) > 0:
            top_outliers = outlier_df.nlargest(min(20, len(outlier_df)), 'out_degree')
            
            plt.figure(figsize=(14, 8))
            colors = ['red' if deleted else 'blue' for deleted in top_outliers['is_deleted']]
            bars = plt.bar(range(len(top_outliers)), top_outliers['out_degree'], 
                          color=colors, alpha=0.7, edgecolor='black')
            
            plt.xlabel('Top Outlier Nodes (Ranked)')
            plt.ylabel('Out-degree')
            plt.title('Top Out-degree Outliers (Red: Deleted, Blue: Active)')
            plt.xticks(range(len(top_outliers)), 
                      [f"ID:{row['node_id']}\nP:{row['original_partition']}" 
                       for _, row in top_outliers.iterrows()], rotation=45)
            plt.grid(True, alpha=0.3)
            
            # 값 표시
            for i, (bar, degree) in enumerate(zip(bars, top_outliers['out_degree'])):
                plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, 
                        str(degree), ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(f"{output_prefix}_top_outliers_detail.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    print(" -> Out-degree 시각화 완료")

def analyze_inter_partition_edges_memory_efficient(
    graph_file_path: str,
    partition_file_path: str,
    output_prefix: str,
    num_partitions: int,
    size_t_bytes: int = 8,
    num_workers: int = 0,
    outlier_percentile: float = 95  # 새로운 매개변수 추가
):
    metadata_format = 'Q' if size_t_bytes == 8 else 'I'

    print(f"'{partition_file_path}'에서 파티션 정보를 로딩합니다...")
    try:
        with open(partition_file_path, 'rb') as f:
            metadata_size = size_t_bytes * 2
            num_points, _ = struct.unpack(f'<{metadata_format}{metadata_format}', f.read(metadata_size))
            
            # --- [수정] 음수 ID를 읽기 위해 int32로 변경 ---
            partition_labels = np.fromfile(f, dtype=np.int32)
            
            if num_points != len(partition_labels):
                raise ValueError(f"파티션 파일 메타데이터({num_points})와 실제 데이터 크기({len(partition_labels)}) 불일치")
            
            print(f"성공: {len(partition_labels)}개 노드에 대한 파티션 정보를 로드했습니다.")
    except Exception as e:
        print(f"오류: 파티션 파일 처리 중 문제 발생. ({e})")
        return

    print(f"'{graph_file_path}'를 사전 스캔하여 각 노드의 위치(offset)를 찾습니다...")
    offsets = np.zeros(num_points, dtype=np.int64)
    total_edge_count = 0
    nodes_in_graph_file = 0
    try:
        with open(graph_file_path, 'rb') as f:
            # DiskANN _disk.graph 헤더 (num_points, max_degree)
            header_size = 16 
            f.seek(header_size)
            current_offset = header_size
            
            for i in tqdm(range(int(num_points)), desc="Pre-scanning graph"):
                offsets[i] = current_offset
                num_neighbors_bytes = f.read(4)
                if not num_neighbors_bytes: break
                
                num_neighbors = struct.unpack('<I', num_neighbors_bytes)[0]
                total_edge_count += num_neighbors
                
                bytes_to_skip = num_neighbors * 4
                current_offset += (4 + bytes_to_skip)
                f.seek(bytes_to_skip, os.SEEK_CUR)
                nodes_in_graph_file += 1
            
        if num_points != nodes_in_graph_file:
                 print(f"경고: 파티션 파일 포인트 수({num_points})와 그래프 파일 노드 수({nodes_in_graph_file})가 다릅니다.")

    except Exception as e:
        print(f"오류: 그래프 파일 사전 스캔 중 문제 발생. ({e})")
        return

    if num_workers == 0:
        num_workers = cpu_count()
    print(f"{num_workers}개의 워커를 사용하여 병렬 분석을 시작합니다...")

    node_indices = np.arange(nodes_in_graph_file)
    chunks = np.array_split(node_indices, num_workers)
    tasks = [(chunk, offsets, graph_file_path, partition_labels, num_partitions) for chunk in chunks]

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.map(process_chunk, tasks), total=len(tasks), desc="Parallel Processing"))

    print("\n모든 워커의 작업 완료. 결과를 취합합니다...")
    
    # 행렬과 out-degree 정보를 분리하여 취합
    matrices = [result[0] for result in results]
    out_degree_dicts = [result[1] for result in results]
    
    inter_edge_matrix = np.sum(matrices, axis=0)
    
    # Out-degree 정보 통합
    all_out_degrees = {}
    for out_degree_dict in out_degree_dicts:
        all_out_degrees.update(out_degree_dict)
    
    # 이중 집계로 인해 행렬의 총합은 실제 엣지 수보다 클 수 있습니다.
    print(f"실제 총 엣지 수 (On-disk Edges): {total_edge_count}")
    print(f"분석된 엣지 수 (Matrix Sum): {np.sum(inter_edge_matrix)}")

    # Out-degree outlier 분석 수행
    outlier_df, degree_stats = analyze_out_degree_outliers(
        all_out_degrees, partition_labels, output_prefix, outlier_percentile
    )

    matrix_labels = [str(i) for i in range(num_partitions)] + ['Deleted']
    counts_matrix_df = pd.DataFrame(inter_edge_matrix, index=matrix_labels, columns=matrix_labels)
    
    # --- 이후 기존 시각화 및 저장 로직은 동일 ---
    if total_edge_count > 0:
        total_percent_df = (counts_matrix_df / total_edge_count) * 100
    else:
        total_percent_df = pd.DataFrame(np.zeros_like(counts_matrix_df), index=matrix_labels, columns=matrix_labels)
        
    row_sum = counts_matrix_df.sum(axis=1)
    # 0으로 나누는 것을 방지
    row_percent_df = counts_matrix_df.div(row_sum[row_sum != 0], axis=0).fillna(0) * 100
    
    output_csv_path = f"{output_prefix}_matrices.csv"
    with open(output_csv_path, 'w') as f:
        f.write("# Edge Counts Matrix (double-counted)\n")
        counts_matrix_df.to_csv(f)
        f.write("\n\n# Total-Edge Percentage Matrix (%)\n")
        total_percent_df.to_csv(f, float_format='%.4f')
        f.write("\n\n# Row-Normalized Percentage Matrix (%)\n")
        row_percent_df.to_csv(f, float_format='%.4f')
    print(f"3개의 분석 결과 행렬이 '{output_csv_path}' 파일에 성공적으로 저장되었습니다.")
    
    print("분석 결과 시각화를 시작합니다...")
    plt.style.use('seaborn-v0_8-whitegrid')
    show_annotations = (num_partitions + 1) <= 20
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(counts_matrix_df.astype(int), annot=show_annotations, fmt='d', cmap='viridis', ax=ax)
    ax.set_title('Edge Counts Matrix (incl. Ghosts of Deleted Nodes)', fontsize=16)
    plt.savefig(f"{output_prefix}_counts_heatmap.png")
    plt.close(fig)
    print(f" -> 카운트 행렬 히트맵 저장 완료.")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(total_percent_df, annot=show_annotations, fmt='.2f', cmap='magma', ax=ax)
    ax.set_title('Total-Edge Percentage Matrix (%)', fontsize=16)
    plt.savefig(f"{output_prefix}_total_percent_heatmap.png")
    plt.close(fig)
    print(f" -> 전체 비율 행렬 히트맵 저장 완료.")

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(row_percent_df, annot=show_annotations, fmt='.1f', cmap='cividis', ax=ax)
    ax.set_title('Row-Normalized Percentage Matrix (%)', fontsize=16)
    plt.savefig(f"{output_prefix}_row_norm_percent_heatmap.png")
    plt.close(fig)
    print(f" -> 행 정규화 비율 행렬 히트맵 저장 완료.")
    
    print("\n모든 분석 및 시각화가 완료되었습니다.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DiskANN 그래프와 파티션 정보를 메모리 효율적인 병렬 방식으로 분석하고 시각화합니다.")
    parser.add_argument('--graph_file', type=str, required=True, help='입력 그래프 파일 경로 (e.g., _disk.graph)')
    parser.add_argument('--partition_file', type=str, required=True, help='파티션 정보 파일 경로 (.partition)')
    parser.add_argument('--output_prefix', type=str, required=True, help='분석 결과를 저장할 파일들의 접두사')
    parser.add_argument('-k', '--num_partitions', type=int, required=True, help='전체 파티션(클러스터)의 수')
    parser.add_argument('--size_t_bytes', type=int, default=8, choices=[4, 8], help='파일을 생성한 C++ 환경의 size_t 타입 크기')
    parser.add_argument('--num_workers', type=int, default=0, help='사용할 워커(프로세스)의 수. 0이면 모든 CPU 코어를 사용합니다.')
    parser.add_argument('--outlier_percentile', type=float, default=95, help='Out-degree outlier 임계값 백분위수 (기본값: 95)')
    args = parser.parse_args()

    analyze_inter_partition_edges_memory_efficient(
        graph_file_path=args.graph_file,
        partition_file_path=args.partition_file,
        output_prefix=args.output_prefix,
        num_partitions=args.num_partitions,
        size_t_bytes=args.size_t_bytes,
        num_workers=args.num_workers,
        outlier_percentile=args.outlier_percentile
    )