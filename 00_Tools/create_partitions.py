import numpy as np
import struct
import argparse
from typing import Literal

# 지원하는 데이터 타입을 정의합니다.
SUPPORTED_DTYPES = {
    'float': np.float32,
    'int8': np.int8,
    'uint8': np.uint8,
}
# 삭제된 노드를 표시하기 위한 ID (모든 스크립트에서 동일한 값 유지)
INVALID_PARTITION_ID = np.iinfo(np.uint32).max

def create_partitions(
    data_file_path: str,
    partition_file_path: str,
    num_partitions: int,
    use_gpu: bool,
    existing_partition_file: str | None = None,
    data_type: Literal['float', 'int8', 'uint8'] = 'float',
    size_t_bytes: int = 8,
    dimension_override: int | None = None,
    centroid_file_path: str | None = None
):
    """
    DiskANN의 .data 파일을 읽어 K-Means 클러스터링을 적용하고 결과를 저장합니다.
    기존 파티션 파일의 삭제 정보를 유지할 수 있습니다.
    """
    if data_type not in SUPPORTED_DTYPES:
        raise ValueError(f"지원하지 않는 데이터 타입입니다. 사용 가능한 타입: {list(SUPPORTED_DTYPES.keys())}")
    
    if size_t_bytes not in [4, 8]:
        raise ValueError("size_t_bytes는 4 또는 8이어야 합니다.")

    dtype = SUPPORTED_DTYPES[data_type]
    metadata_format = 'Q' if size_t_bytes == 8 else 'I'
    metadata_size = size_t_bytes * 2

    print(f"'{data_file_path}' 파일을 읽는 중...", flush=True)
    with open(data_file_path, 'rb') as f:
        num_points, dim_from_file = struct.unpack(f'<{metadata_format}{metadata_format}', f.read(metadata_size))
        dimension = dimension_override if dimension_override is not None else dim_from_file
        all_data = np.fromfile(f, dtype=dtype).reshape(num_points, dimension)
        print(f"데이터 정보: {num_points}개 포인트, {dimension}차원", flush=True)

    if existing_partition_file:
        print(f"'{existing_partition_file}'에서 기존 파티션 정보를 로드하여 삭제된 노드를 식별합니다.", flush=True)
        with open(existing_partition_file, 'rb') as f:
            f.read(metadata_size) # 메타데이터 건너뛰기
            existing_labels = np.fromfile(f, dtype=np.uint32)
        
        # --- START OF MODIFICATION ---
        # 데이터 파일과 파티션 파일의 포인트 수가 1 차이 나는 것을 허용하고, 작은 쪽에 맞춤
        if abs(len(existing_labels) - num_points) > 1:
            raise ValueError(f"데이터 파일({num_points}개)과 기존 파티션 파일({len(existing_labels)}개)의 포인트 수가 너무 많이 차이납니다.")
        elif len(existing_labels) != num_points:
            min_len = min(num_points, len(existing_labels))
            print(f"경고: 데이터 파일과 파티션 파일의 포인트 수가 1 차이납니다. 더 작은 값({min_len})을 기준으로 작업을 계속합니다.", flush=True)
            all_data = all_data[:min_len]
            existing_labels = existing_labels[:min_len]
            num_points = min_len
        # --- END OF MODIFICATION ---
            
        valid_nodes_mask = existing_labels != INVALID_PARTITION_ID
        data_to_cluster = all_data[valid_nodes_mask]
        print(f"총 {num_points}개 중 {len(data_to_cluster)}개의 유효한 포인트로 클러스터링을 수행합니다.", flush=True)
    else:
        valid_nodes_mask = np.ones(num_points, dtype=bool)
        data_to_cluster = all_data
        print("기존 파티션 파일이 없어 모든 포인트를 대상으로 클러스터링을 수행합니다.", flush=True)

    if np.issubdtype(data_to_cluster.dtype, np.integer):
        data_to_cluster = data_to_cluster.astype(np.float32)

    if use_gpu:
        try:
            from cuml.cluster import KMeans as cuMLKMeans
            print(f"{num_partitions}개의 파티션으로 GPU K-Means 클러스터링을 시작합니다...", flush=True)
            kmeans = cuMLKMeans(n_clusters=num_partitions, random_state=42, n_init=10)
        except ImportError:
            use_gpu = False
            
    if not use_gpu:
        from sklearn.cluster import KMeans
        print(f"{num_partitions}개의 파티션으로 CPU K-Means 클러스터링을 시작합니다...", flush=True)
        kmeans = KMeans(n_clusters=num_partitions, random_state=42, n_init=10, verbose=0)

    new_labels_for_valid_nodes = kmeans.fit_predict(data_to_cluster)
    print("클러스터링 완료. 결과 파일을 저장합니다...", flush=True)

    if use_gpu and hasattr(new_labels_for_valid_nodes, 'get'):
        new_labels_for_valid_nodes = new_labels_for_valid_nodes.get()
    
    final_partition_labels = np.full(num_points, INVALID_PARTITION_ID, dtype=np.uint32)
    final_partition_labels[valid_nodes_mask] = new_labels_for_valid_nodes

    with open(partition_file_path, 'wb') as f:
        f.write(struct.pack(f'<{metadata_format}{metadata_format}', num_points, 1))
        final_partition_labels.astype(np.uint32).tofile(f)
    print(f"파티션 정보가 '{partition_file_path}' 파일에 성공적으로 저장되었습니다.", flush=True)

    if centroid_file_path:
        centroids = kmeans.cluster_centers_
        if use_gpu and hasattr(centroids, 'get'):
            centroids = centroids.get()
        with open(centroid_file_path, 'wb') as f:
            f.write(struct.pack(f'<{metadata_format}{metadata_format}', num_partitions, dimension))
            centroids.astype(np.float32).tofile(f)
        print(f"중심점 정보가 '{centroid_file_path}' 파일에 성공적으로 저장되었습니다.", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DiskANN .data 파일에 K-Means 클러스터링을 적용하여 .partition 파일을 생성합니다.")
    
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--partition_file', type=str, required=True)
    parser.add_argument('-k', '--num_partitions', type=int, required=True)
    parser.add_argument('--dtype', type=str, default='float', choices=SUPPORTED_DTYPES.keys())
    parser.add_argument('--size_t_bytes', type=int, default=8, choices=[4, 8])
    parser.add_argument('--dimension', dest='dimension_override', type=int)
    parser.add_argument('--centroid_file', type=str)
    parser.add_argument('--gpu', action='store_true', help='이 플래그를 사용하면 GPU 가속 K-Means를 사용합니다 (cuML 필요).')
    parser.add_argument('--existing_partition_file', type=str, help='(선택) 삭제 정보를 가져올 기존 .partition 파일 경로')
    
    args = parser.parse_args()

    create_partitions(
        data_file_path=args.data_file,
        partition_file_path=args.partition_file,
        num_partitions=args.num_partitions,
        use_gpu=args.gpu,
        existing_partition_file=args.existing_partition_file,
        data_type=args.dtype,
        size_t_bytes=args.size_t_bytes,
        dimension_override=args.dimension_override,
        centroid_file_path=args.centroid_file
    )