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

# datapaths
data='../02_Datasets/sift/sift_learn.fbin'
query='../02_Datasets/sift/sift_query.fbin'

index_prefix='../03_Simulation_data/01_Index_files/sift/index'
result='../03_Simulation_data/05_Simulation_results/sift/res'
index=${index_prefix}.in-memory-index
gt_file=../03_Simulation_data/02_Ground_truth/sift/gt100.in-memory-index
partition_path=${index}.partition



# run_with_log "build_memory_index" \
#     ./../apps/build_memory_index  --data_type float \
#                                 --dist_fn l2 \
#                                 --data_path ${data} \
#                                 --index_path_prefix ${index} \
#                                 -R 32 -L 50 --alpha 1.2


# run_with_log "create_partitions" \
#     python ./../03_Simulation_data/00_Tools/create_partitions.py \
#         --data_file ${index}.data \
#         --partition_file ${partition_path} \
#         --num_partitions 8 \
#         --dtype float \
#         --size_t_bytes 4 \
#         --dimension 128


# run_with_log "compute_groundtruth" \
#     ./../apps/utils/compute_groundtruth  --data_type float \
#                                     --dist_fn l2 \
#                                     --base_file ${data} \
#                                     --query_file ${query} \
#                                     --gt_file ${gt_file} \
#                                     --K 100

run_with_log "search_memory_index_with_trace" \
    ./../apps/search_memory_index_with_trace  --data_type float \
                                --dist_fn l2 \
                                --index_path_prefix ${index} \
                                --query_file ${query} \
                                --gt_file ${gt_file} \
                                -K 10 -L 10 20 30 40 50 100 \
                                --result_path ${result} \
                                --partition_file ${partition_path}
