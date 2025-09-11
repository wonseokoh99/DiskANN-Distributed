# Tests updating scenario
# Every iteration, it updates 30% of dataset and deletes 30% of nodes
# After updates, its outputs are index files(index, data, tag, partition), ground truth
# This simulation is to show if graph index perperties are changing over updates 
#
# UPDATE : build index, and updated nodes are incresing
#
# UPDATE : partitions are not created every iteration anymore




# Dataset
DATASET='sift'

SIM_NAME="03_test_graph_update"

# Datapaths
DATA_PATH="../02_Datasets/${DATASET}/${DATASET}_base.fbin"
QUERY_PATH="../02_Datasets/${DATASET}/${DATASET}_query.fbin"

INDEX_DIR="../03_Simulation_data/01_Index_files/${SIM_NAME}/${DATASET}"
GT_DIR="../03_Simulation_data/02_Ground_truth/${SIM_NAME}/${DATASET}"
RESULT_DIR="../03_Simulation_data/05_Simulation_results/${SIM_NAME}/${DATASET}"

INDEX_PREFIX="${INDEX_DIR}/index"
GT_PREFIX="${GT_DIR}/gt100"
RESULT_PREFIX="${RESULT_DIR}/res"


# SIM_MODIFIER="-from-${start_idx}-to-${end_idx}-delete-${points_to_delete_from_beginning}"
# INDEX_PATH="${INDEX_PREFIX}${SIM_MODIFIER}"
# GT_PATH="="../03_Simulation_data/02_Fround_truth/sift/gt100${SIM_MODIFIER}"
# PARTITION_PATH="${INDEX_PREFIX}.partition"
# RESULT_PATH

BUILD_DIR="../apps"
UTILS_DIR="${BUILD_DIR}/utils"
TOOLS_DIR="../03_Simulation_data/00_Tools"


# Simulation Parameters
type='float'
dist_fn='l2'
points_to_skip=0
beginning_index_size=600000                # 1M
pts_per_checkpoint=10000                   # 100K
max_points_to_insert=600001                # 1M + 300K 
points_to_delete_from_beginning=30000      # 300K
deletes_after=0

thr=64
norm=1

iteration=1
interval=30000

num_partitions=8


#!/bin/bash
if [ -z "$BASH_VERSION" ]; then
    exec bash "$0" "$@"
fi


# Move to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1


# Logging
LOG_DIR=../03_Simulation_data/10_Logs/${DATASET}
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

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


# MKDIR
mkdir -p ${INDEX_DIR}
mkdir -p ${GT_DIR}
mkdir -p ${RESULT_DIR}



# Simulation


# Initial graph build
run_with_log "build_continuous_memory_index" \
    ${BUILD_DIR}/build_continuous_memory_index  --data_type ${type} \
                                                --index_path_prefix ${INDEX_PREFIX} \
                                                --dist_fn ${dist_fn} \
                                                --data_path ${DATA_PATH} \
                                                --points_to_skip 0 \
                                                --beginning_index_size ${beginning_index_size} \
                                                --points_per_checkpoint ${pts_per_checkpoint} \
                                                --checkpoints_per_snapshot 0 \
                                                --points_to_delete_from_beginning 0 \
                                                -R 32 -L 100 --alpha 1.2 -T ${thr} \
                                                --max_points_to_insert ${max_points_to_insert} \
                                                --start_deletes_after 0 \
                                                --do_concurrent true --start_point_norm 1;

SIM_MODIFIER="-from-${points_to_skip}-to-${end_idx}-delete-${points_to_delete_from_beginning}"
INDEX_PATH="${INDEX_PREFIX}${SIM_MODIFIER}"
PARTITION_PATH="${INDEX_PATH}.partition${num_partitions}"
CENTROID_PATH="${INDEX_PATH}.centroids${num_partitions}"


run_with_log "create_partitions" \
    python ${TOOLS_DIR}/create_partitions.py --data_file ${INDEX_PATH}.data \
                                    --partition_file ${PARTITION_PATH} \
                                    --num_partitions ${num_partitions} \
                                    --centroid_file  ${CENTROID_PATH} \
                                    --dtype ${type} \
                                    --size_t_bytes 4 \
                                    --dimension 128


while [ $iteration -le 10 ]; do

end_idx=$((points_to_skip + max_points_to_insert))
SIM_MODIFIER_BEFORE=${SIM_MODIFIER}
SIM_MODIFIER="-from-${points_to_skip}-to-${end_idx}-delete-${points_to_delete_from_beginning}"

INDEX_PATH="${INDEX_PREFIX}${SIM_MODIFIER}"
GT_PATH="${GT_PREFIX}${SIM_MODIFIER}"
PARTITION_PATH_OLD="${INDEX_PREFIX}${SIM_MODIFIER_OLD}.partition${num_partitions}"
PARTITION_PATH="${INDEX_PATH}.partition${num_partitions}"
CENTROID_PATH="${INDEX_PATH}.centroids${num_partitions}"
RESULT_PATH="${RESULT_PREFIX}${SIM_MODIFIER}"

it=$((iteration * interval))

echo "\n-----iteration: $iteration-----"

# mkdir -p "${INDEX_DIR}"
run_with_log "build_continuous_memory_index" \
    ${BUILD_DIR}/build_continuous_memory_index  --data_type ${type} \
                                                --index_path_prefix ${INDEX_PREFIX} \
                                                --dist_fn ${dist_fn} \
                                                --data_path ${DATA_PATH} \
                                                --points_to_skip ${points_to_skip} \
                                                --beginning_index_size ${beginning_index_size} \
                                                --points_per_checkpoint ${pts_per_checkpoint} \
                                                --checkpoints_per_snapshot 0 \
                                                --points_to_delete_from_beginning ${points_to_delete_from_beginning} \
                                                -R 32 -L 100 --alpha 1.2 -T ${thr} \
                                                --max_points_to_insert ${beginning_index_size} \
                                                --start_deletes_after ${deletes_after} \
                                                --do_concurrent true --start_point_norm 1;



run_with_log "update_partitions" \
    python ${TOOLS_DIR}/update_partitions.py    --partition_file_in ${PARTITION_PATH_OLD} \
                                                --partition_file_out ${PARTITION_PATH} \
                                                --data_file  ${INDEX_PATH}.data  \
                                                --centroid_file ${CENTROID_PATH} \
                                                --insert_offset ${beginning_index_size} \
                                                --insert_count ${it} \
                                                --delete_offset 0 \
                                                --delete_count ${it} \
                                                --insert_strategy closest_centroid


run_with_log "compute_groundtruth" \
    ${UTILS_DIR}/compute_groundtruth --data_type ${type} \
                                    --dist_fn ${dist_fn} \
                                    --base_file ${INDEX_PATH}.data  \
                                    --query_file ${QUERY_PATH}  \
                                    --K 100 --gt_file ${GT_PATH} \
                                    #--tags_file  ${index_prefix}.tags


mkdir -p "${RESULT_PATH}"
run_with_log "search_memory_index_with_trace" \
    ${BUILD_DIR}/search_memory_index_with_trace  --data_type ${type} \
                                                --dist_fn ${dist_fn} \
                                                --index_path_prefix ${INDEX_PATH} \
                                                --query_file ${QUERY_PATH} \
                                                --gt_file ${GT_PATH} \
                                                -K 10 -L 10 20 30 40 50 100 \
                                                --result_path ${RESULT_PATH}/res \
                                                --partition_file ${PARTITION_PATH}


points_to_delete_from_beginning=$((points_to_delete_from_beginning + iteration * interval))
max_points_to_insert=$((max_points_to_insert + iteration * interval))
    iteration=$((iteration + 1))  
done



