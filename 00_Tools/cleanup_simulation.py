import os
import shutil
import argparse
import glob



# dry run
# python cleanup_simulation.py --sim_data_dir ../03_Simulation_data/04_build_partition_graph/sift
#
# actual run 
# python cleanup_simulation.py --sim_data_dir ../03_Simulation_data/04_build_partition_graph/sift --run
#
 






def cleanup_simulation_files(sim_data_dir: str, do_run: bool):
    """
    시뮬레이션 결과로 생성된 파일 및 디렉터리를 정리합니다.
    기본적으로는 Dry Run으로 삭제될 대상만 보여줍니다.

    Args:
        sim_data_dir (str): 정리할 시뮬레이션 데이터의 상위 디렉터리 경로.
                            (예: ../03_Simulation_data/04_build_partition_graph/sift)
        do_run (bool): True일 경우 실제로 파일을 삭제합니다.
    """
    
    print(f"'{sim_data_dir}' 경로의 시뮬레이션 결과물을 정리합니다.")
    
    # 정리 대상 경로 정의
    index_dir = os.path.join(sim_data_dir, "01_Index_files")
    gt_dir = os.path.join(sim_data_dir, "02_Ground_truth")
    result_dir = os.path.join(sim_data_dir, "05_Simulation_results")
    
    # 분석 결과 디렉터리 경로 (사용자 스크립트에 따라 경로가 다를 수 있으므로 확인 필요)
    # 예시: ../04_Simulation_analysis/04_build_partition_graph/sift
    analysis_dir_parent = os.path.abspath(os.path.join(sim_data_dir, "../../../04_Simulation_analysis"))
    sim_name = os.path.basename(os.path.abspath(os.path.join(sim_data_dir, "..")))
    dataset_name = os.path.basename(sim_data_dir)
    analysis_dir = os.path.join(analysis_dir_parent, sim_name, dataset_name)
    
    
    # 삭제할 파일 및 디렉터리 목록 수집
    files_to_delete = []
    dirs_to_delete = []

    # 1. 반복(iteration)을 통해 생성된 파일 및 디렉터리 찾기
    # '-from-' 패턴이 들어간 모든 파일을 대상으로 합니다.
    
    # Index, Partition, Centroids 파일
    if os.path.exists(index_dir):
        files_to_delete.extend(glob.glob(os.path.join(index_dir, "*-from-*")))
    
    # Ground Truth 파일
    if os.path.exists(gt_dir):
        files_to_delete.extend(glob.glob(os.path.join(gt_dir, "*-from-*")))
        
    # Result 디렉터리 (iter_ 또는 from-to- 스타일)
    if os.path.exists(result_dir):
        dirs_to_delete.extend(glob.glob(os.path.join(result_dir, "iter_*")))
        dirs_to_delete.extend(glob.glob(os.path.join(result_dir, "from-*")))

    # Analysis 디렉터리
    if os.path.exists(analysis_dir):
        dirs_to_delete.extend(glob.glob(os.path.join(analysis_dir, "from-*")))

    if not files_to_delete and not dirs_to_delete:
        print("정리할 파일이나 디렉터리가 없습니다.")
        return

    print("\n--- 삭제 대상 목록 ---")
    for d in dirs_to_delete:
        print(f"[디렉터리] {d}")
    for f in files_to_delete:
        print(f"[파일]     {f}")
    print("----------------------\n")

    # 2. 실제 삭제 실행 ( --run 옵션이 있을 경우)
    if do_run:
        confirm = input("위의 모든 파일과 디렉터리를 영구적으로 삭제하시겠습니까? (yes/no): ")
        if confirm.lower() == 'yes':
            print("삭제를 실행합니다...")
            for d in tqdm(dirs_to_delete, desc="Deleting Dirs"):
                try:
                    shutil.rmtree(d)
                except OSError as e:
                    print(f"오류: 디렉터리 {d} 삭제 실패 - {e}")
            
            for f in tqdm(files_to_delete, desc="Deleting Files"):
                try:
                    os.remove(f)
                except OSError as e:
                    print(f"오류: 파일 {f} 삭제 실패 - {e}")
            print("정리가 완료되었습니다.")
        else:
            print("삭제를 취소했습니다.")
    else:
        print("Dry Run 모드로 실행되었습니다. 실제 파일은 삭제되지 않았습니다.")
        print("실제로 삭제하려면 --run 플래그를 추가하여 다시 실행해주세요.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DiskANN 시뮬레이션 결과 파일들을 정리합니다."
    )
    parser.add_argument(
        '--sim_data_dir', type=str, required=True,
        help='정리할 시뮬레이션 데이터의 상위 디렉터리. (예: ../03_Simulation_data/04_build_partition_graph/sift)'
    )
    parser.add_argument(
        '--run', action='store_true',
        help='이 플래그를 추가하면 실제로 파일을 삭제합니다. 없으면 삭제 대상만 보여줍니다.'
    )
    
    # tqdm은 필수가 아니므로, 없어도 동작하도록 처리
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, **kwargs):
            return iterable

    args = parser.parse_args()
    cleanup_simulation_files(args.sim_data_dir, args.run)


    