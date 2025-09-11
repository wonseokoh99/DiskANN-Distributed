import numpy as np
import pandas as pd
import struct
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def analyze_degree_distribution(
    graph_file_path: str,
    output_prefix: str,
    size_t_bytes: int = 8
):
    """
    DiskANN 그래프 파일을 읽어 각 노드의 In/Out Degree를 계산하고,
    통계 및 시각화 자료를 생성합니다.
    """
    if size_t_bytes not in [4, 8]:
        raise ValueError("size_t_bytes는 4 또는 8이어야 합니다.")

    print(f"'{graph_file_path}' 그래프 파일을 분석합니다...")

    try:
        with open(graph_file_path, 'rb') as f:
            # --- 1. 그래프 파일 헤더 읽기 ---
            size_t_format = 'Q' if size_t_bytes == 8 else 'I'
            header_format = f'<{size_t_format}II{size_t_format}'
            header_size = struct.calcsize(header_format)
            
            header_bytes = f.read(header_size)
            if len(header_bytes) < header_size:
                raise IOError("그래프 파일이 너무 작아 헤더를 읽을 수 없습니다.")

            file_size, max_degree, start_node, num_frozen = struct.unpack(header_format, header_bytes)
            print(f"그래프 헤더 정보: Max Degree={max_degree}, Start Node={start_node}, Frozen Points={num_frozen}")
            if max_degree == 0:
                print("경고: 파일 헤더에서 Max Degree를 0으로 읽었습니다. size_t_bytes 옵션이 올바른지 확인해주세요.")

            # --- 2. In/Out Degree 계산 ---
            in_degrees_list = [0] * 1
            out_degrees = []
            
            pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Analyzing Graph File")
            pbar.update(header_size)

            while True:
                num_neighbors_bytes = f.read(4)
                if not num_neighbors_bytes: break
                
                num_neighbors = struct.unpack('<I', num_neighbors_bytes)[0]
                out_degrees.append(num_neighbors)

                if num_neighbors > 0:
                    neighbors_bytes_size = num_neighbors * 4
                    neighbors = np.fromfile(f, dtype=np.uint32, count=num_neighbors)
                    pbar.update(4 + neighbors_bytes_size)
                    
                    for neighbor_id in neighbors:
                        if neighbor_id >= len(in_degrees_list):
                            in_degrees_list.extend([0] * (neighbor_id - len(in_degrees_list) + 1))
                        in_degrees_list[neighbor_id] += 1
                else:
                    pbar.update(4)
            
            pbar.close()
            num_nodes = len(out_degrees)
            if len(in_degrees_list) < num_nodes:
                in_degrees_list.extend([0] * (num_nodes - len(in_degrees_list)))
            in_degrees = np.array(in_degrees_list[:num_nodes], dtype=np.uint32)
            print(f"총 {num_nodes}개의 노드에 대한 분석 완료.")

    except (IOError, ValueError, FileNotFoundError) as e:
        print(f"오류: 파일 처리 중 문제가 발생했습니다. ({e})")
        return

    # 3. 통계 데이터프레임 생성
    degree_df = pd.DataFrame({
        'node_id': np.arange(num_nodes),
        'out_degree': out_degrees,
        'in_degree': in_degrees
    })

    stats_filename = f"{output_prefix}_degree_stats.csv"
    degree_df.to_csv(stats_filename, index=False)
    print(f" -> 각 노드의 Degree 정보가 '{stats_filename}'에 저장되었습니다.")

    print("\n--- Degree 분포 요약 통계 ---")
    print(degree_df[['out_degree', 'in_degree']].describe())
    print("---------------------------\n")

    # 4. Degree 분포 시각화
    print("Degree 분포를 시각화합니다...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Out-Degree 분포
    sns.histplot(data=degree_df, x='out_degree', bins=max(20, int(degree_df['out_degree'].max())), kde=True, ax=axes[0])
    axes[0].set_title('Out-Degree Distribution', fontsize=16)
    axes[0].set_xlabel('Out-Degree (Edges from Node)', fontsize=12)
    axes[0].set_ylabel('Number of Nodes', fontsize=12)

    # In-Degree 분포 (상위 1% 제외)
    # --- START OF MODIFICATION ---
    if not degree_df['in_degree'].empty:
        # 99번째 백분위수 값을 계산하여 상위 1%의 경계값을 찾습니다.
        p99 = degree_df['in_degree'].quantile(0.99)
        print(f"In-Degree의 상위 1%에 해당하는 값(경계값): {p99:.2f}")
        
        # 경계값보다 작은 데이터만 필터링하여 시각화에 사용합니다.
        in_degree_filtered = degree_df[degree_df['in_degree'] <= p99]
        
        sns.histplot(data=in_degree_filtered, x='in_degree', kde=True, ax=axes[1], color='salmon')
        axes[1].set_title('In-Degree Distribution (Bottom 99%)', fontsize=16)
    else:
        axes[1].set_title('In-Degree Distribution (No Data)', fontsize=16)

    axes[1].set_xlabel('In-Degree (Edges to Node)', fontsize=12)
    # --- END OF MODIFICATION ---
    
    hist_filename = f"{output_prefix}_degree_distribution.png"
    plt.suptitle('Graph Degree Distribution Analysis', fontsize=20, y=1.02)
    plt.tight_layout()
    plt.savefig(hist_filename)
    plt.close(fig)

    print(f" -> Degree 분포 그래프가 '{hist_filename}'에 저장되었습니다.")
    print("분석이 완료되었습니다.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DiskANN 그래프 파일의 In/Out Degree 분포를 분석하고 시각화합니다."
    )
    parser.add_argument('--graph_file', type=str, required=True)
    parser.add_argument('--output_prefix', type=str, required=True)
    parser.add_argument('--size_t_bytes', type=int, default=8, choices=[4, 8])
    args = parser.parse_args()
    
    analyze_degree_distribution(
        graph_file_path=args.graph_file,
        output_prefix=args.output_prefix,
        size_t_bytes=args.size_t_bytes
    )