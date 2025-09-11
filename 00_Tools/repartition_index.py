import numpy as np
import struct
import argparse
from typing import Literal

def load_data(data_file_path: str, size_t_bytes: int, dtype: np.dtype, dimension: int | None):
    """ .data 파일을 로드하고 메타데이터와 데이터를 반환합니다. """
    metadata_format = 'Q' if size_t_bytes == 8 else 'I'
    metadata_size = size_t_bytes * 2
    with open(data_file_path, 'rb') as f:
        num_points, dim_from_file = struct.unpack(f'<{metadata_format}{metadata_format}', f.read(metadata_size))
        if dimension is None:
            dimension = dim_from_file
        
        data = np.fromfile(f, dtype=dtype).reshape(num_points, dimension)
        print(f"데이터 파일 로드 완료: {num_points}개 포인트, {dimension}차원", flush=True)
        return data, num_points, dimension

def repartition_index(
    data_file_path: str,
    existing_partition_file: str,
    output_partition_file: str,
    num_partitions: int,
    use_gpu: bool,
    data_type: str,
    size_t_bytes: int,
    dimension: int | None,
    output_centroid_file: str | None
):
    """
    기존 파티션 파일의 삭제 정보(음수 ID)를 유지하면서, 유효한 데이터에 대해서만
    K-Means 클러스터링을 다시 수행합니다.
    """
    dtype = {'float': np.float32, 'int8': np.int8, 'uint8': np.uint8}[data_type]
    metadata_format = 'Q' if size_t_bytes == 8 else 'I'
    metadata_size = size_t_bytes * 2

    # 1. 기존 파티션 파일 로드 (음수 ID 처리를 위해 int32로 로드)
    print(f"'{existing_partition_file}'에서 기존 파티션 정보를 로드하여 삭제된 노드를 식별합니다.", flush=True)
    with open(existing_partition_file, 'rb') as f:
        num_total_slots, _ = struct.unpack(f'<{metadata_format}{metadata_format}', f.read(metadata_size))
        # --- [수정 1] 데이터 타입을 int32로 변경 ---
        existing_labels = np.fromfile(f, dtype=np.int32)
    
    # 2. 데이터 파일 로드
    all_data, num_points_in_data, dimension = load_data(data_file_path, size_t_bytes, dtype, dimension)
    
    # 3. 데이터 크기 검증 및 조정
    # --- [수정 2] 유효 노드(0 또는 양수 ID)를 기준으로 마스크 생성 ---
    valid_nodes_mask = existing_labels >= 0
    num_valid_nodes = np.sum(valid_nodes_mask)

    if abs(num_points_in_data - num_valid_nodes) > 1:
        raise ValueError(f"데이터 파일의 포인트 수({num_points_in_data})와 "
                         f"기존 파티션의 유효한 노드 수({num_valid_nodes})가 너무 많이 차이납니다.")
    
    if num_points_in_data != num_valid_nodes:
        min_len = min(num_points_in_data, num_valid_nodes)
        print(f"경고: 데이터 파일과 유효한 노드 수가 1 차이납니다. 더 작은 값({min_len})을 기준으로 작업을 계속합니다.")
        
        data_to_cluster = all_data[:min_len]
        
        true_indices = np.where(valid_nodes_mask)[0]
        indices_to_keep = true_indices[:min_len]
        valid_nodes_mask = np.zeros_like(existing_labels, dtype=bool)
        valid_nodes_mask[indices_to_keep] = True
    else:
        data_to_cluster = all_data

    print(f"총 {num_total_slots}개 슬롯 중 {len(data_to_cluster)}개의 유효한 포인트로 클러스터링을 수행합니다.", flush=True)

    if np.issubdtype(data_to_cluster.dtype, np.integer):
        data_to_cluster = data_to_cluster.astype(np.float32)
    
    # 4. K-Means 클러스터링 실행 (로직 동일)
    kmeans = None
    if use_gpu:
        try:
            from cuml.cluster import KMeans as cuMLKMeans
            print("GPU K-Means를 시작합니다...", flush=True)
            kmeans = cuMLKMeans(n_clusters=num_partitions, random_state=42, n_init=10)
        except ImportError:
            use_gpu = False
            print("오류: cuML을 찾을 수 없어 CPU 모드로 전환합니다.", flush=True)
    
    if not use_gpu:
        from sklearn.cluster import KMeans
        print("CPU K-Means를 시작합니다...", flush=True)
        kmeans = KMeans(n_clusters=num_partitions, random_state=42, n_init=10, verbose=0)

    new_labels_for_valid_nodes = kmeans.fit_predict(data_to_cluster)
    print("클러스터링 완료.", flush=True)

    if use_gpu and hasattr(new_labels_for_valid_nodes, 'get'):
        new_labels_for_valid_nodes = new_labels_for_valid_nodes.get()

    # --- [수정 3] 최종 파티션 생성 방식 변경 ---
    # 5. 최종 파티션 파일 생성 (삭제 정보 보존)
    # 기존 파티션 정보(특히 삭제된 노드의 음수 ID)를 그대로 복사합니다.
    final_partition_labels = existing_labels.copy().astype(np.int32)
    # 유효한 노드들의 위치에만 새로 계산된 K-Means 레이블을 덮어씁니다.
    final_partition_labels[valid_nodes_mask] = new_labels_for_valid_nodes.astype(np.int32)
    
    with open(output_partition_file, 'wb') as f:
        f.write(struct.pack(f'<{metadata_format}{metadata_format}', num_total_slots, 1))
        final_partition_labels.tofile(f)
    print(f"새로운 파티션 파일이 '{output_partition_file}'에 저장되었습니다.", flush=True)

    if output_centroid_file:
        centroids = kmeans.cluster_centers_
        if use_gpu and hasattr(centroids, 'get'):
            centroids = centroids.get()
        with open(output_centroid_file, 'wb') as f:
            f.write(struct.pack(f'<{metadata_format}{metadata_format}', num_partitions, dimension))
            centroids.astype(np.float32).tofile(f)
        print(f"새로운 중심점 파일이 '{output_centroid_file}'에 저장되었습니다.", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DiskANN 인덱스 파일을 읽어 재파티셔닝을 수행합니다.")
    parser.add_argument('--data_file', type=str, required=True, help='클러스터링할 데이터가 담긴 .data 파일 (압축된 상태)')
    parser.add_argument('--existing_partition_file', type=str, required=True, help='삭제 정보를 가져올 기존 .partition 파일')
    parser.add_argument('--output_partition_file', type=str, required=True, help='결과를 저장할 새로운 .partition 파일 경로')
    parser.add_argument('-k', '--num_partitions', type=int, required=True, help='K-Means에 사용할 파티션(클러스터) 수')
    
    parser.add_argument('--gpu', action='store_true', help='GPU 가속 K-Means를 사용합니다 (cuML 필요).')
    parser.add_argument('--dtype', type=str, default='float', choices=['float', 'int8', 'uint8'])
    parser.add_argument('--size_t_bytes', type=int, default=8, choices=[4, 8])
    parser.add_argument('--dimension', type=int, help='벡터 차원 수 (선택 사항, .data 파일에서 읽음)')
    parser.add_argument('--output_centroid_file', type=str, help='(선택) 계산된 중심점을 저장할 파일 경로')

    args = parser.parse_args()
    
    repartition_index(
        data_file_path=args.data_file,
        existing_partition_file=args.existing_partition_file,
        output_partition_file=args.output_partition_file,
        num_partitions=args.num_partitions,
        use_gpu=args.gpu,
        data_type=args.dtype,
        size_t_bytes=args.size_t_bytes,
        dimension=args.dimension,
        output_centroid_file=args.output_centroid_file
    )