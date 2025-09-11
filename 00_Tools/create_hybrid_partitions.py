import numpy as np
import pandas as pd
import struct
import argparse
from tqdm import tqdm

# --- Helper Functions ---
def load_partition_file(file_path, size_t_bytes):
    metadata_format = 'Q' if size_t_bytes == 8 else 'I'
    with open(file_path, 'rb') as f:
        metadata_size = size_t_bytes * 2
        num_points, _ = struct.unpack(f'<{metadata_format}{metadata_format}', f.read(metadata_size))
        labels = np.fromfile(f, dtype=np.int32)
        if len(labels) != num_points:
            raise ValueError(f"Partition file {file_path} is corrupt.")
        return labels

def load_data_file(file_path, dtype, size_t_bytes):
    metadata_format = 'Q' if size_t_bytes == 8 else 'I'
    dtype_map = {'float': np.float32, 'int8': np.int8, 'uint8': np.uint8}
    with open(file_path, 'rb') as f:
        metadata_size = size_t_bytes * 2
        num_points, dim = struct.unpack(f'<{metadata_format}{metadata_format}', f.read(metadata_size))
        data = np.fromfile(f, dtype=dtype_map[dtype]).reshape(num_points, dim)
        return data, dim

# --- Main Logic ---
def create_hybrid_partitions(
    partition_file_in_old: str,
    centroid_file_in_old: str,
    data_file_new: str,
    partition_file_out: str,
    centroid_file_out: str,
    delete_count: int,
    insert_count: int,
    size_t_bytes: int,
    dtype: str,
    num_partitions: int,
    gpu: bool  # --- [수정] 누락되었던 gpu 파라미터를 추가했습니다. ---
):
    """Compacted 그래프에 맞춰 파티션 ID를 보존하고, 새 중심점까지 계산하는 하이브리드 스크립트"""
    
    print("--- 하이브리드 파티션 및 중심점 생성 시작 ---")
    
    # 1. 원본(Old) 정보 로드
    partition_labels_old = load_partition_file(partition_file_in_old, size_t_bytes)
    centroids_old, _ = load_data_file(centroid_file_in_old, 'float', size_t_bytes)

    # 2. 신규(New) 데이터 로드
    all_vectors_new, dim = load_data_file(data_file_new, dtype, size_t_bytes)
    num_total_new = len(all_vectors_new)
    
    # 3. 파라미터 계산
    num_survived = num_total_new - insert_count
    
    # 4. 새로운 파티션 배열 생성 및 채우기
    partition_labels_new = np.zeros(num_total_new, dtype=np.int32)

    # 4a. 살아남은 노드 파티션 ID 보존
    for new_id in tqdm(range(num_survived), desc="Preserving Partitions"):
        old_id = new_id + delete_count
        if old_id < len(partition_labels_old):
            partition_labels_new[new_id] = partition_labels_old[old_id]
        else:
            partition_labels_new[new_id] = 0

    # 4b. 신규 노드 파티션 할당
    new_vectors = all_vectors_new[num_survived:]
    for i in tqdm(range(insert_count), desc="Assigning New Partitions"):
        new_id = num_survived + i
        vector = new_vectors[i].astype(np.float32)
        distances = np.linalg.norm(centroids_old - vector, axis=1)
        assigned_partition = np.argmin(distances)
        partition_labels_new[new_id] = assigned_partition
        
    # 5. 하이브리드 파티션 파일 저장
    metadata_format = 'Q' if size_t_bytes == 8 else 'I'
    with open(partition_file_out, 'wb') as f:
        f.write(struct.pack(f'<{metadata_format}{metadata_format}', num_total_new, 1))
        partition_labels_new.tofile(f)
    print(f"하이브리드 파티션 파일 저장 완료: '{partition_file_out}'")

    # 6. 새로운 중심점 계산 및 저장
    print("\n새로운 중심점을 계산합니다...")
    valid_mask = partition_labels_new >= 0
    valid_vectors = all_vectors_new[valid_mask].astype(np.float32)
    valid_labels = partition_labels_new[valid_mask]

    # --- [수정] 함수 내부에서 변수명을 gpu로 사용 ---
    if gpu:
        try:
            import cudf
            print("GPU(cuDF)를 사용하여 중심점 계산을 가속합니다.")
            gdf = cudf.DataFrame(valid_vectors)
            gdf['partition'] = valid_labels
            centroids_df = gdf.groupby('partition').mean().sort_index()
            centroids = centroids_df.to_numpy()
        except ImportError:
            print("cuDF를 찾을 수 없어 CPU 모드로 전환합니다.")
            gpu = False # fallback to CPU

    if not gpu:
        print("CPU(Pandas)를 사용하여 중심점을 계산합니다.")
        df = pd.DataFrame(valid_vectors)
        df['partition'] = valid_labels
        centroids_df = df.groupby('partition').mean().sort_index()
        centroids = centroids_df.to_numpy()
    
    # 모든 파티션이 존재하는지 확인하고, 비어있으면 0벡터로 채움
    if len(centroids) < num_partitions:
        full_centroids = np.zeros((num_partitions, dim), dtype=np.float32)
        existing_partitions = centroids_df.index.to_numpy()
        full_centroids[existing_partitions] = centroids
        centroids = full_centroids

    with open(centroid_file_out, 'wb') as f:
        f.write(struct.pack(f'<{metadata_format}{metadata_format}', num_partitions, dim))
        centroids.astype(np.float32).tofile(f)
    print(f"새로운 중심점 파일 저장 완료: '{centroid_file_out}'")
    print("--- 모든 작업 완료 ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compaction 이후 ID가 보존된 하이브리드 파티션과 새 중심점을 생성합니다.")
    parser.add_argument('--partition-file-in-old', required=True)
    parser.add_argument('--centroid-file-in-old', required=True)
    parser.add_argument('--data-file-new', required=True)
    parser.add_argument('--partition-file-out', required=True)
    parser.add_argument('--centroid-file-out', required=True)
    parser.add_argument('--delete-count', type=int, required=True)
    parser.add_argument('--insert-count', type=int, required=True)
    parser.add_argument('--num-partitions', type=int, required=True)
    parser.add_argument('--size-t-bytes', type=int, default=8, choices=[4, 8])
    parser.add_argument('--dtype', type=str, default='float', choices=['float', 'int8', 'uint8'])
    parser.add_argument('--gpu', action='store_true', help="GPU(cuDF)를 사용하여 중심점 계산을 가속합니다.")

    args = parser.parse_args()
    create_hybrid_partitions(**vars(args))