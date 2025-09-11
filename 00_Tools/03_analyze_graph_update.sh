#!/bin/bash
if [ -z "$BASH_VERSION" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

LOG_DIR="../10_Logs/sift"
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

DATASET="sift"

SIM_NAME="02_test_graph_update"

INDEX_DIR="../01_Index_files/${SIM_NAME}/${DATASET}"
GT_DIR="../02_Ground_truth/${SIM_NAME}/${DATASET}"
RESULT_DIR="../05_Simulation_results/${SIM_NAME}/${DATASET}"

INDEX_PREFIX="${INDEX_DIR}/index"
GT_PREFIX="${GT_DIR}/gt100"
RESULT_PREFIX="${RESULT_DIR}/res"

GRAPH_PREFIX="${INDEX_PREFIX}"
TRACE_PREFIX="${RESULT_DIR}" 
OUT_PREFIX="../06_Analysis/${SIM_NAME}" 

NUM_PARTITIONS=8


# Simulation Parameters
type='float'
dist_fn='l2'
points_to_skip=0
beginning_index_size=600000                # 1M
pts_per_checkpoint=10000                   # 100K
max_points_to_insert=630000                # 1M + 300K 
points_to_delete_from_beginning=30000      # 300K
deletes_after=0

iteration=0
interval=30000


mkdir -p ${OUT_PREFIX}

# Simulation
while [ $iteration -le 7 ]; do


end_idx=$((points_to_skip + max_points_to_insert))

SIM_MODIFIER="-from-${points_to_skip}-to-${end_idx}-delete-${points_to_delete_from_beginning}"

INDEX_PATH="${INDEX_PREFIX}${SIM_MODIFIER}"
GT_PATH="${GT_PREFIX}${SIM_MODIFIER}"
PARTITION_PATH="${INDEX_PATH}.partition${num_partitions}"
RESULT_PATH="${RESULT_PREFIX}${SIM_MODIFIER}"

GRAPH_PATH="${GRAPH_PREFIX}${SIM_MODIFIER}"
PARTITION_PATH="${GRAPH_PATH}.partition${NUM_PARTITIONS}"
TRACE_PATH="${TRACE_PREFIX}/res${SIM_MODIFIER}/res_trace.csv"

OUT_EDGE_DIR="${OUT_PREFIX}/01_Inter_node_edge/res${SIM_MODIFIER}" 
OUT_TRACE_DIR="${OUT_PREFIX}/02_Trace/res${SIM_MODIFIER}" 

mkdir -p ${OUT_EDGE_DIR}
run_with_log "analyze_edges" \
    python analyze_edges.py --graph_file ${GRAPH_PATH} \
                            --partition_file ${PARTITION_PATH} \
                            --output_prefix  ${OUT_EDGE_DIR}/inter_edge \
                            --num_partitions ${NUM_PARTITIONS} \

# mkdir -p ${OUT_TRACE_DIR}
# run_with_log "analyze_trace" \
#     python analyze_trace.py --trace_file ${TRACE_PATH} \
#                             --output_prefix ${OUT_TRACE_DIR}/trace






points_to_delete_from_beginning=$((points_to_delete_from_beginning + iteration * interval))
max_points_to_insert=$((max_points_to_insert + iteration * interval))
    iteration=$((iteration + 1))  
done