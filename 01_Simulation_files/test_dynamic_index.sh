#!/bin/bash
if [ -z "$BASH_VERSION" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

LOG_DIR="../03_Simulation_data/10_Logs/sift"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 진행 상황을 화면에서도 보고 파일에도 저장
run_with_log() {
    local step_name="$1"
    local log_file="${LOG_DIR}/${step_name}_${TIMESTAMP}.log"
    shift
    
    echo "=== $step_name 시작 - $(date) ==="
    
    # exit code를 제대로 캐치하는 방법
    set -o pipefail
    "$@" 2>&1 | tee "$log_file"
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "=== $step_name 완료 ==="
    else
        echo "=== $step_name 실패 (exit code: $exit_code) ==="
        exit $exit_code
    fi
    echo ""
}

# function parameters
type='float'
dist_fn='l2'

deletes=25000
inserts=75000
deletes_after=50000
pts_per_checkpoint=10000
begin=100
thr=64

# datapaths
data='../02_Datasets/sift/sift_base.fbin'
query='../02_Datasets/sift/sift_query.fbin'

index_prefix='../03_Simulation_data/01_Index_files/sift/index'
result='../03_Simulation_data/05_Simulation_results/sift/res'
index=${index_prefix}.after-concurrent-delete-del${deletes}-${inserts}
gt_file=../03_Simulation_data/02_Ground_truth/sift/gt100_learn-conc-${deletes}-${inserts}


run_with_log "index_build" \
    ./../apps/test_insert_deletes_consolidate --data_type ${type} \
                                            --index_path_prefix ${index_prefix} \
                                            --dist_fn ${dist_fn} \
                                            --data_path ${data} \
                                            --points_to_skip 0 \
                                            --beginning_index_size ${begin} \
                                            --points_per_checkpoint ${pts_per_checkpoint} \
                                            --checkpoints_per_snapshot 0 \
                                            --points_to_delete_from_beginning ${deletes} \
                                            -R 64 -L 300 --alpha 1.2 -T ${thr} \
                                            --max_points_to_insert ${inserts} \
                                            --start_deletes_after ${deletes_after} \
                                            --do_concurrent true --start_point_norm 1

run_with_log "compute_gt" \
    ./../apps/utils/compute_groundtruth --data_type ${type} \
                                    --dist_fn l2 --base_file ${index}.data  \
                                    --query_file ${query}  --K 100 --gt_file ${gt_file} \
                                    --tags_file  ${index}.tags

run_with_log "index_search" \
    ./../apps/search_memory_index --data_type ${type} \
                                --dist_fn l2 --index_path_prefix ${index} \
                                --result_path ${result} --query_file ${query}  --gt_file ${gt_file}  \
                                -K 10 -L 20 40 60 80 100 -T ${thr} --dynamic true --tags 1