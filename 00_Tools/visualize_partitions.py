import numpy as np
import pandas as pd
import struct
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_partition_distribution(
    partition_file_path: str,
    output_image_path: str,
    size_t_bytes: int = 8
):
    """
    .partition 파일을 읽어 파티션별 Active/Deleted 노드 분포를 계산하고
    누적 막대그래프로 시각화합니다.

    - Active Node: 파티션 ID가 양수인 노드
    - Deleted Node: 파티션 ID가 음수인 노드 (-(원래 파티션 ID + 1))

    Args:
        partition_file_path (str): 분석할 .partition 파일의 경로.
        output_image_path (str): 생성될 그래프 이미지의 저장 경로.
        size_t_bytes (int): .partition 파일을 생성한 C++ 환경의 size_t 크기 (4 또는 8).
    """
    metadata_format = 'Q' if size_t_bytes == 8 else 'I'

    # 1. 파티션 파일 로드 (음수 ID 처리를 위해 int32로 로드)
    print(f"'{partition_file_path}'에서 파티션 정보를 로딩합니다...")
    try:
        with open(partition_file_path, 'rb') as f:
            metadata_size = size_t_bytes * 2
            metadata_bytes = f.read(metadata_size)
            if len(metadata_bytes) < metadata_size:
                raise IOError("파일이 너무 작아 메타데이터를 읽을 수 없습니다.")
            
            num_points, _ = struct.unpack(f'<{metadata_format}{metadata_format}', metadata_bytes)
            # 음수 값을 읽기 위해 dtype을 int32로 변경
            partition_labels = np.fromfile(f, dtype=np.int32)
            
            if num_points != len(partition_labels):
                raise ValueError(f"파일 메타데이터(포인트 수: {num_points})와 실제 데이터 크기({len(partition_labels)})가 일치하지 않습니다.")
        print(f"총 {len(partition_labels)}개의 노드 정보를 로드했습니다.")
    except (IOError, ValueError, FileNotFoundError) as e:
        print(f"오류: 파일 처리 중 문제가 발생했습니다. ({e})")
        return

    # 2. 파티션별 Active/Deleted 노드 수 계산
    print("파티션별 Active/Deleted 노드 수를 계산합니다...")
    
    if len(partition_labels) == 0:
        print("경고: 파티션 정보가 없습니다. 빈 이미지를 생성합니다.")
        # 빈 이미지 생성 로직은 시각화 부분에서 처리됩니다.
        counts_df = pd.DataFrame(columns=['partition_id', 'Active', 'Deleted'])
    else:
        df = pd.DataFrame(partition_labels, columns=['label'])
        
        # 'status' 열 추가: 양수면 'Active', 음수면 'Deleted'
        df['status'] = np.where(df['label'] >= 0, 'Active', 'Deleted')
        
        # 'partition_id' 열 추가: 음수 레이블을 원래의 양수 파티션 ID로 변환
        # 예: -1 -> 0, -2 -> 1, 0 -> 0, 1 -> 1
        df['partition_id'] = np.where(
            df['label'] >= 0,
            df['label'],
            -df['label'] - 1 
        )
        
        # pivot_table을 사용하여 각 파티션 ID별로 Active, Deleted 노드 수 집계
        counts_df = df.pivot_table(
            index='partition_id',
            columns='status',
            aggfunc='size',
            fill_value=0
        )

        # 결과에 'Active' 또는 'Deleted' 열이 없을 경우 추가
        if 'Active' not in counts_df.columns:
            counts_df['Active'] = 0
        if 'Deleted' not in counts_df.columns:
            counts_df['Deleted'] = 0

        counts_df = counts_df.reset_index()

    print("--- 파티션별 포인트 수 (Active/Deleted) ---")
    print(counts_df)
    print("------------------------------------------\n")

    # 3. 누적 막대그래프 시각화
    print(f"결과를 '{output_image_path}'에 그래프로 저장합니다...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 9))

    # 베이스가 되는 'Active' 노드 막대그래프
    sns.barplot(
        data=counts_df, x='partition_id', y='Active',
        color='royalblue', label='Active', ax=ax
    )
    
    # 'Active' 막대 위에 'Deleted' 노드 막대그래프를 쌓음
    sns.barplot(
        data=counts_df, x='partition_id', y='Deleted',
        bottom=counts_df['Active'],  # Active 막대 위에서 시작하도록 설정
        color='darkorange', label='Deleted', ax=ax
    )

    ax.set_title('Distribution of Active & Deleted Points per Partition', fontsize=18)
    ax.set_xlabel('Partition ID', fontsize=14)
    ax.set_ylabel('Number of Points', fontsize=14)
    ax.legend()

    # x축 레이블이 많을 경우 겹치지 않도록 조정
    if len(counts_df) > 20:
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_image_path)
    plt.close(fig)

    print("시각화가 완료되었습니다.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DiskANN .partition 파일을 읽어 파티션별 Active/Deleted 노드 분포를 누적 막대그래프로 시각화합니다."
    )
    parser.add_argument(
        '--partition_file', type=str, required=True,
        help='분석할 .partition 파일의 경로'
    )
    parser.add_argument(
        '--output_image', type=str, required=True,
        help='생성될 막대그래프 이미지 파일의 저장 경로 (예: distribution.png)'
    )
    parser.add_argument(
        '--size_t_bytes', type=int, default=8, choices=[4, 8],
        help='바이너리 파일의 size_t 타입 크기 (기본값: 8)'
    )
    args = parser.parse_args()
    
    visualize_partition_distribution(
        partition_file_path=args.partition_file,
        output_image_path=args.output_image,
        size_t_bytes=args.size_t_bytes
    )