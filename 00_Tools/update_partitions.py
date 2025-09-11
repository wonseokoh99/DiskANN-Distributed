import numpy as np
import struct
import argparse
from typing import Literal
from tqdm import tqdm

# 지원하는 데이터 타입을 정의합니다.
SUPPORTED_DTYPES = {
    'float': np.float32,
    'int8': np.int8,
    'uint8': np.uint8,
}
# --- START OF MODIFICATION ---
# 파티션 레이블을 저장할 데이터 타입을 부호 있는 정수로 변경
PARTITION_DTYPE = np.int32
# --- END OF MODIFICATION ---

def load_binary_file(file_path: str, dtype: np.dtype, size_t_bytes: int):
    metadata_format = 'Q' if size_t_bytes == 8 else 'I'
    with open(file_path, 'rb') as f:
        metadata_size = size_t_bytes * 2
        num_points, dim = struct.unpack(f'<{metadata_format}{metadata_format}', f.read(metadata_size))
        data = np.fromfile(f, dtype=dtype).reshape(num_points, dim)
        return data

def update_partitions(
    partition_file_in: str,
    partition_file_out: str,
    size_t_bytes: int,
    data_file: str | None = None,
    insert_offset: int | None = None,
    insert_count: int | None = None,
    centroid_file_in: str | None = None,
    centroid_file_out: str | None = None,
    delete_offset: int | None = None,
    delete_count: int | None = None,
    data_type: Literal['float', 'int8', 'uint8'] = 'float',
    insert_strategy: Literal['closest_centroid', 'smallest_partition'] = 'closest_centroid',
    update_centroids: bool = False
):
    metadata_format = 'Q' if size_t_bytes == 8 else 'I'
    
    # 1. 기존 파티션 파일 로드 (dtype 수정)
    print(f"기존 파티션 파일 '{partition_file_in}'을 로드합니다...")
    with open(partition_file_in, 'rb') as f:
        metadata_size = size_t_bytes * 2
        num_points, _ = struct.unpack(f'<{metadata_format}{metadata_format}', f.read(metadata_size))
        # --- dtype을 PARTITION_DTYPE (int32)으로 변경 ---
        partition_labels = np.fromfile(f, dtype=PARTITION_DTYPE)
    
    if len(partition_labels) != num_points:
        raise ValueError("입력 파티션 파일의 메타데이터와 실제 크기가 다릅니다.")
    print(f"{num_points}개의 파티션 정보를 로드했습니다.")

    # 2. Delete 연산 수행 (음수화 로직으로 변경)
    if delete_offset is not None and delete_count is not None:
        print(f"오프셋 {delete_offset}부터 {delete_count}개의 노드를 삭제 처리합니다...")
        start, end = delete_offset, delete_offset + delete_count
        delete_ids = np.arange(start, min(end, len(partition_labels)))
        
        # --- START OF MODIFICATION ---
        # 0 이상의 유효한 파티션 ID만 음수로 변경 (중복 삭제 방지)
        for node_id in delete_ids:
            if partition_labels[node_id] >= 0:
                partition_labels[node_id] = -(partition_labels[node_id] + 1)
        # --- END OF MODIFICATION ---
        print(f"{len(delete_ids)}개 노드의 파티션을 '삭제됨(음수)'으로 처리했습니다.")

    # 3. Insert 연산 수행
    if insert_offset is not None and insert_count is not None:
        print(f"오프셋 {insert_offset}부터 {insert_count}개의 노드에 대해 '{insert_strategy}' 전략으로 파티션을 할당합니다...")
        
        if not data_file:
            raise ValueError("--insert_offset을 사용하려면 --data_file이 반드시 필요합니다.")

        # --- 유효한 파티션만 카운트하도록 수정 ---
        valid_labels = partition_labels[partition_labels >= 0]
        num_partitions = np.max(valid_labels) + 1 if len(valid_labels) > 0 else 1
        partition_counts = np.bincount(valid_labels, minlength=num_partitions)
        print(f"현재 파티션별 노드 수: {partition_counts}")

        if insert_strategy == 'closest_centroid' or update_centroids:
            if not centroid_file_in:
                raise ValueError("'closest_centroid' 전략이나 --update_centroids 옵션을 사용하려면 --centroid_file_in이 필요합니다.")
            centroids = load_binary_file(centroid_file_in, np.float32, size_t_bytes)

        print(f"'{data_file}'에서 전체 벡터 데이터를 로드합니다...")
        all_vectors = load_binary_file(data_file, SUPPORTED_DTYPES[data_type], size_t_bytes)
        new_vectors = all_vectors[insert_offset : insert_offset + insert_count]
        
        max_id = insert_offset + insert_count - 1
        if max_id >= len(partition_labels):
            new_size = max_id + 1
            # --- 새 슬롯을 -1 (삭제된 파티션 0)로 초기화 ---
            resized_labels = np.full(new_size, -1, dtype=PARTITION_DTYPE)
            resized_labels[:len(partition_labels)] = partition_labels
            partition_labels = resized_labels

        for i in tqdm(range(insert_count), desc="Assigning Partitions"):
            node_id = insert_offset + i
            vector = new_vectors[i].astype(np.float32)
            
            if insert_strategy == 'closest_centroid':
                distances = np.linalg.norm(centroids - vector, axis=1)
                assigned_partition = np.argmin(distances)
            else: # smallest_partition
                assigned_partition = np.argmin(partition_counts)
            
            partition_labels[node_id] = assigned_partition

            if update_centroids:
                n = partition_counts[assigned_partition]
                c_old = centroids[assigned_partition]
                # n=0인 경우 분모가 0이 되는 것을 방지
                c_new = (c_old * n + vector) / (n + 1) if n > 0 else vector
                centroids[assigned_partition] = c_new
            
            partition_counts[assigned_partition] += 1
        
        print(f"{insert_count}개의 새로운 노드에 파티션을 할당했습니다.")

    # 4. 업데이트된 파티션 파일 저장
    print(f"업데이트된 파티션 정보를 '{partition_file_out}'에 저장합니다...")
    with open(partition_file_out, 'wb') as f:
        f.write(struct.pack(f'<{metadata_format}{metadata_format}', len(partition_labels), 1))
        partition_labels.tofile(f) # int32 배열이 저장됨
    
    if update_centroids and centroid_file_out:        
        print(f"업데이트된 중심점 정보를 '{centroid_file_out}'에 저장합니다...")
        with open(centroid_file_out, 'wb') as f:
            num_centroids, dim = centroids.shape
            f.write(struct.pack(f'<{metadata_format}{metadata_format}', num_centroids, dim))
            centroids.tofile(f)

    print("작업이 완료되었습니다.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="기존 .partition 파일에 오프셋 기반 insert/delete 연산을 적용하고, 선택적으로 centroid를 업데이트합니다.")
    parser.add_argument('--partition_file_in', type=str, required=True)
    parser.add_argument('--partition_file_out', type=str, required=True)
    
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--insert_offset', type=int)
    parser.add_argument('--insert_count', type=int)
    parser.add_argument('--delete_offset', type=int)
    parser.add_argument('--delete_count', type=int)
    
    # --- Centroid 관련 인자 수정 ---
    parser.add_argument('--centroid_file_in', type=str, help='입력으로 사용할 K-Means 중심점 파일')
    parser.add_argument('--centroid_file_out', type=str, help='(선택) 업데이트된 중심점을 저장할 파일')
    parser.add_argument('--update_centroids', action='store_true', help='(선택) 이 플래그를 사용하면 Insert 시 중심점을 업데이트하고 저장합니다.')

    parser.add_argument('--insert_strategy', type=str, default='closest_centroid', choices=['closest_centroid', 'smallest_partition'])
    parser.add_argument('--size_t_bytes', type=int, default=8, choices=[4, 8])
    parser.add_argument('--dtype', type=str, default='float', choices=SUPPORTED_DTYPES.keys())

    args = parser.parse_args()

    # 인자 유효성 검증
    if args.update_centroids and not args.centroid_file_out:
        parser.error("--update_centroids를 사용하려면 --centroid_file_out도 함께 지정해야 합니다.")

    update_partitions(
        partition_file_in=args.partition_file_in,
        partition_file_out=args.partition_file_out,
        size_t_bytes=args.size_t_bytes,
        data_file=args.data_file,
        insert_offset=args.insert_offset,
        insert_count=args.insert_count,
        centroid_file_in=args.centroid_file_in,
        centroid_file_out=args.centroid_file_out,
        delete_offset=args.delete_offset,
        delete_count=args.delete_count,
        data_type=args.dtype,
        insert_strategy=args.insert_strategy,
        update_centroids=args.update_centroids
    )