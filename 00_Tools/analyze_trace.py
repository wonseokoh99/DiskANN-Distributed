import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import numpy as np

def analyze_detailed_trace(trace_file: str, output_prefix: str):
    """
    상세 정보가 포함된 trace.csv 파일을 분석하여 다양한 통계를 추출하고 시각화합니다.
    음수 partition_id를 가진 삭제된 노드 기록은 분석에서 제외합니다.
    """
    print(f"'{trace_file}' 파일을 로딩합니다...")
    try:
        df = pd.read_csv(trace_file)
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다 - {trace_file}")
        return

    # --- 삭제된 노드 필터링 ---
    print(f"분석 전, 총 {len(df)}개의 트레이스 기록 로드됨.")
    df_valid = df[df['partition_id'] >= 0].copy()
    print(f"삭제된 노드 관련 기록 제외 후, {len(df_valid)}개의 기록으로 분석을 시작합니다...")

    if df_valid.empty:
        print("경고: 유효한 트레이스 기록이 없습니다. 분석을 중단합니다.")
        return

    if 'L' not in df_valid.columns:
        df_valid['L'] = 'all_data'
    else:
        # L 값을 순서가 있는 카테고리형으로 변환하여 그래프 순서 고정
        df_valid['L'] = pd.Categorical(df_valid['L'], ordered=True, categories=sorted(df_valid['L'].unique()))
    
    print("파일 로딩 완료. 분석을 시작합니다...")

    # ================================================================
    # 데이터 분석 파트
    # ================================================================

    # 1. 파티션 이동 정보 계산
    print("1. 파티션 이동 정보 계산 중...")
    df_valid['next_partition_id'] = df_valid.groupby(['query_id', 'L'])['partition_id'].shift(-1)
    transitions = df_valid.dropna(subset=['next_partition_id']).copy()
    transitions = transitions[transitions['next_partition_id'] >= 0]
    transitions['next_partition_id'] = transitions['next_partition_id'].astype(int)
    transitions['transition_type'] = np.where(
        transitions['partition_id'] == transitions['next_partition_id'], 
        'Intra-Partition', 
        'Inter-Partition'
    )
    
    # 2a. 쿼리별 파티션 이동 타입 분석
    print("2a. 쿼리별 파티션 이동 타입 분석 중...")
    query_transition_counts = transitions.groupby(['query_id', 'L', 'transition_type']).size().unstack(fill_value=0).reset_index()
    if 'Intra-Partition' not in query_transition_counts.columns:
        query_transition_counts['Intra-Partition'] = 0
    if 'Inter-Partition' not in query_transition_counts.columns:
        query_transition_counts['Inter-Partition'] = 0
    query_transition_counts.to_csv(f"{output_prefix}_query_transition_counts.csv", index=False)
    print(f" -> 각 쿼리별 파티션 이동 횟수가 저장되었습니다.")
    print("\n--- L값별 평균 파티션 이동 횟수 ---")
    print(query_transition_counts.groupby('L')[['Intra-Partition', 'Inter-Partition']].mean())
    print("----------------------------------\n")

    # 2b. 쿼리별 통계 요약 (거리계산, 경로길이)
    print("2b. 쿼리별 통계 요약 중...")
    query_summary = df_valid.groupby(['query_id', 'L']).agg(
        total_intra_comps=('intra_partition_comps', 'sum'),
        total_inter_comps=('inter_partition_comps', 'sum'),
        max_hop=('hop', 'max')
    ).reset_index()
    query_summary['path_length'] = query_summary['max_hop'] + 1
    query_summary.to_csv(f"{output_prefix}_query_summary.csv", index=False)
    print(f" -> 각 쿼리별 통계가 저장되었습니다.")
    print("\n--- L값별 평균 쿼리 통계 ---")
    print(query_summary.groupby('L').agg(
        avg_path_length=('path_length', 'mean'),
        avg_intra_comps=('total_intra_comps', 'mean'),
        avg_inter_comps=('total_inter_comps', 'mean')
    ))
    print("----------------------------\n")

    # 3. 파티션 방문 빈도 계산
    print("3. 파티션 방문 빈도 계산 중...")
    partition_visits = df_valid['partition_id'].value_counts().reset_index()
    partition_visits.columns = ['partition_id', 'visit_count']
    partition_visits.to_csv(f"{output_prefix}_partition_visits.csv", index=False)
    print(f" -> 파티션 방문 빈도 정보가 저장되었습니다.")

    # 4. Hop별 탐색 동향 분석
    print("4. Hop별 탐색 동향 분석 중...")
    hop_dynamics = df_valid.groupby(['L', 'hop']).agg(
        avg_intra_comps=('intra_partition_comps', 'mean'),
        avg_inter_comps=('inter_partition_comps', 'mean')
    ).reset_index()
    hop_dynamics.to_csv(f"{output_prefix}_hop_dynamics.csv", index=False)
    print(f" -> Hop별 탐색 동향 정보가 저장되었습니다.")

    # ================================================================
    # 시각화 파트
    # ================================================================
    print("분석 결과 시각화를 시작합니다...")
    plt.style.use('seaborn-v0_8-whitegrid')

    # 시각화 1: Hop별 탐색 동향 (선 그래프)
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.lineplot(data=hop_dynamics, x='hop', y='avg_intra_comps', hue='L', palette='winter', ax=ax, legend='full', lw=2)
    sns.lineplot(data=hop_dynamics, x='hop', y='avg_inter_comps', hue='L', palette='autumn', ax=ax, legend=False, linestyle='--', lw=2)
    ax.set_title('Search Dynamics per Hop (Intra vs Inter Partition Compares)', fontsize=16)
    ax.set_xlabel('Hop', fontsize=12)
    ax.set_ylabel('Average Distance Computations', fontsize=12)
    ax.legend(title='L value (Solid: Intra, Dashed: Inter)')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    plt.savefig(f"{output_prefix}_hop_dynamics.png")
    plt.close(fig)
    print(f" -> Hop별 탐색 동향 그래프가 저장되었습니다.")

    # 시각화 2: 탐색 경로 길이 분포 (KDE 그래프)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.kdeplot(data=query_summary, x='path_length', hue='L', palette='viridis', fill=True, common_norm=False, ax=ax)
    ax.set_title('Distribution of Search Path Lengths by L value')
    ax.set_xlabel('Search Path Length (Number of Hops)')
    ax.set_ylabel('Density')
    plt.savefig(f"{output_prefix}_path_length_distribution.png")
    plt.close(fig)
    print(f" -> 경로 길이 분포 그래프가 저장되었습니다.")
    
    # 시각화 3: 전체 파티션 이동 매트릭스 (히트맵)
    total_transitions = transitions.groupby(['partition_id', 'next_partition_id']).size().reset_index(name='count')
    transition_matrix = total_transitions.pivot_table(index='partition_id', columns='next_partition_id', values='count', fill_value=0)
    transition_matrix.to_csv(f"{output_prefix}_transition_matrix.csv")
    fig, ax = plt.subplots(figsize=(12, 10))
    show_annotations = len(transition_matrix) <= 20
    sns.heatmap(transition_matrix.astype(int), annot=show_annotations, fmt='d', cmap='viridis', ax=ax)
    ax.set_title('Overall Partition Transition Heatmap')
    ax.set_xlabel('To Partition')
    ax.set_ylabel('From Partition')
    plt.savefig(f"{output_prefix}_transition_matrix_heatmap.png")
    plt.close(fig)
    print(f" -> 이동 매트릭스 히트맵이 저장되었습니다.")

    # 시각화 4: L값별 파티션 이동 횟수 분포 (Hop 기준 Box Plot)
    melted_transitions = query_transition_counts.melt(
        id_vars=['L'], 
        value_vars=['Intra-Partition', 'Inter-Partition'],
        var_name='Transition Type', 
        value_name='Count'
    )
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=melted_transitions, x='L', y='Count', hue='Transition Type', ax=ax)
    ax.set_title('Distribution of Transition Counts per Query by L value')
    ax.set_xlabel('L value')
    ax.set_ylabel('Number of Transitions (Hops)')
    ax.legend(title='Transition Type')
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_transition_counts_distribution.png")
    plt.close(fig)
    print(f" -> 파티션 이동 횟수(Hop) 분포 그래프가 저장되었습니다.")

    # --- [추가] 시각화 5: L값별 거리 계산 횟수 분포 (Distance Comps 기준 Box Plot) ---
    print("추가: L값별 거리 계산 횟수 분포 시각화 중...")
    melted_comps = query_summary.melt(
        id_vars=['L'], 
        value_vars=['total_intra_comps', 'total_inter_comps'],
        var_name='Comparison Type', 
        value_name='Count'
    )
    # 보기 좋은 레이블을 위해 이름 변경
    melted_comps['Comparison Type'] = melted_comps['Comparison Type'].replace({
        'total_intra_comps': 'Intra-Partition Comps',
        'total_inter_comps': 'Inter-Partition Comps'
    })

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=melted_comps, x='L', y='Count', hue='Comparison Type', ax=ax)
    ax.set_title('Distribution of Distance Computations per Query by L value')
    ax.set_xlabel('L value')
    ax.set_ylabel('Number of Distance Computations')
    ax.legend(title='Comparison Type')
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_distance_comps_distribution.png")
    plt.close(fig)
    print(f" -> 거리 계산 횟수 분포 그래프가 저장되었습니다.")


    print("\n모든 분석 및 시각화가 완료되었습니다.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="상세 정보가 포함된 DiskANN search trace.csv 파일을 분석합니다.")
    parser.add_argument('--trace_file', type=str, required=True, help='분석할 trace.csv 파일의 경로')
    parser.add_argument('--output_prefix', type=str, required=True, help='저장될 결과 파일들의 이름 앞에 붙일 접두사')
    args = parser.parse_args()
    
    pd.set_option('display.width', 120)
    
    analyze_detailed_trace(args.trace_file, args.output_prefix)