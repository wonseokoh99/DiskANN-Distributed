# 
# Simulation to check if re-clustering and re-build 
# improves inter-node edge count and  
# cluster imbalance
# 


#!/bin/bash
if [ -z "$BASH_VERSION" ]; then
    exec bash "$0" "$@"
fi

# --- [수정] 오류 발생 시 즉시 중단 옵션 추가 ---
set -e
# ----------------------------------------------


# Move to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# ================================================================
# Simulation Parameters (설정값을 한 곳에서 관리)
# ================================================================
TYPE=float
DIST_FN=l2
POINTS_TO_DELETE=100000
POINTS_TO_INSERT=0
START_POINT=0
INITIAL_INDEX_SIZE=900000
NUM_PARTITION=4
DIMENSION=128
MAX_ITERATIONS=8


DATA_SIZE=4
# ================================================================
# Path and Directory Setup
# ================================================================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DATASET='sift'
SIM_NAME="04_build_partition_graph-smallest_partition"
SIM_PATH_MODIFIER="${SIM_NAME}-${DATASET}-${TIMESTAMP}"

# Main simulation directory that includes the timestamp
SIM_DIR="../03_Simulation_data/${SIM_PATH_MODIFIER}"
INDEX_DIR="${SIM_DIR}/01_Index_files"
GT_DIR="${SIM_DIR}/02_Ground_truth"
RESULT_DIR="${SIM_DIR}/05_Simulation_results"
ANALYSIS_DIR="${SIM_DIR}/06_Simulation_analysis/"
LOG_DIR="${SIM_DIR}/10_Log"

INDEX_PREFIX="${INDEX_DIR}/index"
GT_PREFIX="${GT_DIR}/gt100"

BUILD_DIR="../apps"
UTILS_DIR="${BUILD_DIR}/utils"
TOOLS_DIR="../00_Tools"

# MKDIR
mkdir -p ${INDEX_DIR} ${GT_DIR} ${RESULT_DIR} ${ANALYSIS_DIR} ${LOG_DIR}


# Datapaths
DATA_PATH="../02_Datasets/${DATASET}/${DATASET}_base.fbin"
QUERY_PATH="../02_Datasets/${DATASET}/${DATASET}_query.fbin"

# ================================================================
# Helper Functions
# ================================================================

# --- [추가] 실패 시 디렉터리 이름 변경을 위한 함수 및 트랩 ---
handle_failure() {
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "!!! 스크립트 실행 실패: 오류가 발생했습니다."
    echo "!!! 로그 파일을 확인해주세요."
    echo "!!! 디렉터리 이름을 ${SIM_DIR}_FAILED 로 변경합니다."
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    # SIM_DIR 변수가 설정되고 디렉터리가 실제로 존재할 때만 실행
    if [ -n "${SIM_DIR}" ] && [ -d "${SIM_DIR}" ]; then
        mv "${SIM_DIR}" "${SIM_DIR}_FAILED"
    fi
    # 실패했음을 명시적으로 알리며 종료
    exit 1
}
# -----------------------------------------------------------

# Logging
# 로그 파일 순번을 위한 카운터 초기화
LOG_STEP_COUNTER=0

run_with_log() {
    local step_name="$1"
    local step_prefix=$(printf "%02d" $LOG_STEP_COUNTER)
    local log_file="${LOG_DIR}/${step_prefix}_${step_name}_${TIMESTAMP}.log"
    LOG_STEP_COUNTER=$((LOG_STEP_COUNTER + 1))
    shift
    
    echo -e "\n=== ${step_prefix} $step_name 시작 - $(date) ==="
    
    # 명령어를 서브셸에서 실행하여 pipefail을 지역적으로 적용하고, 실패 여부를 직접 확인합니다.
    # 'if ! (명령어)' 구문은 명령어가 실패했을 때(non-zero exit code) then 블록을 실행합니다.
    if ! (set -o pipefail; "$@" 2>&1 | tee "$log_file"); then
        # 실패 시, handle_failure 함수를 직접 호출합니다.
        handle_failure
    fi
    
    echo "=== ${step_prefix} $step_name 완료 ==="
    echo ""
}

# 설정을 파일로 저장하는 함수
save_config() {
    local config_file="$1/00_simulation_config.txt"
    echo "Saving simulation config to ${config_file}"
    # Heredoc을 사용하여 여러 줄을 파일에 쉽게 씁니다.
    cat > "${config_file}" << EOL
# Simulation Configuration from run at ${TIMESTAMP}

[Simulation Parameters]
DATASET=${DATASET}
SIM_NAME=${SIM_NAME}
TYPE=${TYPE}
DIST_FN=${DIST_FN}
POINTS_TO_DELETE=${POINTS_TO_DELETE}
POINTS_TO_INSERT=${POINTS_TO_INSERT}
START_POINT=${START_POINT}
INITIAL_INDEX_SIZE=${INITIAL_INDEX_SIZE}
NUM_PARTITION=${NUM_PARTITION}
DIMENSION=${DIMENSION}
MAX_ITERATIONS=${MAX_ITERATIONS}

[Paths]
DATA_PATH=${DATA_PATH}
QUERY_PATH=${QUERY_PATH}
SIMULATION_ROOT_DIR=${SIM_DIR}
EOL
}



# ================================================================
# 스크립트 실행 시작
# ================================================================

# --- 스크립트 시작 시 설정 저장 ---
save_config "${SIM_DIR}"

# Simulation Parameters

CURRENT_START_POINT="${START_POINT}";
CURRENT_INDEX_SIZE=${INITIAL_INDEX_SIZE}

DO_CONC="true"


# ================================================================
# Phase 1 : Initial graph build
# ================================================================
# echo "---------- PHASE 1 : Initial graph build ----------"

# --- Phase 1 경로 정의 ---
PATH_MODIFIER="-from-${CURRENT_START_POINT}-to-$((CURRENT_START_POINT + CURRENT_INDEX_SIZE))"
INDEX_PATH="${INDEX_PREFIX}${PATH_MODIFIER}"
PARTITION_PATH="${INDEX_PATH}.partition${NUM_PARTITION}"
CENTROID_PATH="${INDEX_PATH}.centroids${NUM_PARTITION}"
GT_PATH="${GT_PREFIX}${PATH_MODIFIER}"

run_with_log "build_memory_index_offset_initial" \
    ${BUILD_DIR}/build_memory_index_offset --data_type ${TYPE} \
                                          --dist_fn ${DIST_FN} \
                                          --index_path_prefix ${INDEX_PATH} \
                                          --data_path ${DATA_PATH} \
                                          --points_to_skip ${START_POINT} \
                                          --beginning_index_size ${INITIAL_INDEX_SIZE} \
                                          -R 64 -L 100

run_with_log "create_partitions_initial" \
  conda run -n rapids python -u ${TOOLS_DIR}/create_partitions.py --data_file ${INDEX_PATH}.data \
                                            --partition_file ${PARTITION_PATH} \
                                            --num_partitions ${NUM_PARTITION} \
                                            --centroid_file  ${CENTROID_PATH} \
                                            --dtype ${TYPE} \
                                            --size_t_bytes ${DATA_SIZE} \
                                            --dimension ${DIMENSION} \
                                            --gpu

run_with_log "compute_groundtruth_initial" \
    ${UTILS_DIR}/compute_groundtruth --data_type ${TYPE} \
                                    --dist_fn ${DIST_FN} \
                                    --base_file ${INDEX_PATH}.data  \
                                    --query_file ${QUERY_PATH}  \
                                    --K 100 --gt_file ${GT_PATH}

# --- Phase 1 검색 결과 경로 정의 및 생성 ---
DIR_MODIFIER="${PATH_MODIFIER#-}"
RESULT_DIR_INITIAL="${RESULT_DIR}/${DIR_MODIFIER}"
mkdir -p "${RESULT_DIR_INITIAL}"
RESULT_PATH_PREFIX_INITIAL="${RESULT_DIR_INITIAL}/res"

run_with_log "search_memory_index_with_trace_initial" \
    ${BUILD_DIR}/search_memory_index_with_trace  --data_type ${TYPE} \
                                                --dist_fn ${DIST_FN} \
                                                --index_path_prefix ${INDEX_PATH} \
                                                --query_file ${QUERY_PATH} \
                                                --gt_file ${GT_PATH} \
                                                -K 10 -L 10 20 30 40 50 100 \
                                                --result_path ${RESULT_PATH_PREFIX_INITIAL} \
                                                --partition_file ${PARTITION_PATH}

    ANALYSIS_DIR_INIT="${ANALYSIS_DIR}/${DIR_MODIFIER}"
    mkdir -p "${ANALYSIS_DIR_INIT}"
    run_with_log "analyze_all" \
        conda run -n rapids python -u ${TOOLS_DIR}/visualize_partitions.py \
                                            --partition_file ${PARTITION_PATH} \
                                            --output_image ${ANALYSIS_DIR_INIT}/points_analysis \
                                            --size_t_bytes ${DATA_SIZE} && \
        conda run -n rapids python -u ${TOOLS_DIR}/analyze_edges.py --graph_file ${INDEX_PATH} \
                                             --partition_file ${PARTITION_PATH} \
                                             --output_prefix ${ANALYSIS_DIR_INIT}/edge_analysis \
                                             --num_partitions ${NUM_PARTITION} \
                                             --size_t_bytes ${DATA_SIZE} && \
        conda run -n rapids python -u ${TOOLS_DIR}/analyze_trace.py --trace_file ${RESULT_DIR_INITIAL}/res_trace.csv \
                                             --output_prefix ${ANALYSIS_DIR_INIT}/trace_analysis && \
        conda run -n rapids python -u ${TOOLS_DIR}/analyze_degree.py \
                                            --graph_file ${INDEX_PATH} \
                                            --output_prefix ${ANALYSIS_DIR_INIT}/degree_analysis \
                                            --size_t_bytes ${DATA_SIZE}





# ================================================================
# Phase 2 : Iterative Update (Compacted with Hybrid Partition)
# ================================================================
echo "---------- PHASE 2 : Updated graph build (Compacted) ----------"

# 루프 시작 전, Phase 1에서 생성된 초기 경로를 "이전 경로"로 설정합니다.
PATH_MODIFIER_OLD="-from-${CURRENT_START_POINT}-to-$((CURRENT_START_POINT + CURRENT_INDEX_SIZE))"
INDEX_PATH_OLD="${INDEX_PREFIX}${PATH_MODIFIER_OLD}"
iteration=0
while [ $iteration -lt ${MAX_ITERATIONS} ]; do
    iteration=$((iteration + 1))
    echo "---------- Iteration ${iteration} (Incremental Update) ----------"

    # --- 1. 다음 단계의 변수들 계산 ---
    NEXT_START_POINT=$((CURRENT_START_POINT + POINTS_TO_DELETE))
    NEXT_INDEX_SIZE=$((CURRENT_INDEX_SIZE + POINTS_TO_INSERT - POINTS_TO_DELETE))
    
    PATH_MODIFIER_NEW="-from-${NEXT_START_POINT}-to-$((NEXT_START_POINT + NEXT_INDEX_SIZE))"
    INDEX_PATH_NEW="${INDEX_PREFIX}${PATH_MODIFIER_NEW}"

    # --- 2. C++ update_memory_index 실행 ---
    run_with_log "update_memory_index_iter_${iteration}" \
        ${BUILD_DIR}/update_memory_index      --data_type ${TYPE} \
                                              --dist_fn ${DIST_FN} \
                                              --data_path ${DATA_PATH} \
                                              --index_path_prefix ${INDEX_PATH_OLD} \
                                              --index_save_path ${INDEX_PREFIX} \
                                              --input_index_size ${CURRENT_INDEX_SIZE} \
                                              --points_to_skip ${CURRENT_START_POINT} \
                                              --points_to_delete_from_beginning ${POINTS_TO_DELETE} \
                                              --points_to_insert ${POINTS_TO_INSERT} \
                                              --do_concurrent ${DO_CONC} \
                                              -R 64 -L 100 --alpha 1.2
    
    # --- 3. [수정] 통합된 create_hybrid_partitions.py 한번만 실행 ---
    PARTITION_PATH_OLD="${INDEX_PATH_OLD}.partition${NUM_PARTITION}"
    CENTROID_PATH_OLD="${INDEX_PATH_OLD}.centroids${NUM_PARTITION}"
    PARTITION_PATH_NEW="${INDEX_PATH_NEW}.partition${NUM_PARTITION}"
    CENTROID_PATH_NEW="${INDEX_PATH_NEW}.centroids${NUM_PARTITION}"
    
    run_with_log "create_hybrid_partitions_and_centroids_iter_${iteration}" \
        conda run -n rapids python ${TOOLS_DIR}/create_hybrid_partitions.py \
            --partition-file-in-old ${PARTITION_PATH_OLD} \
            --centroid-file-in-old ${CENTROID_PATH_OLD} \
            --data-file-new ${INDEX_PATH_NEW}.data \
            --partition-file-out ${PARTITION_PATH_NEW} \
            --centroid-file-out ${CENTROID_PATH_NEW} \
            --delete-count ${POINTS_TO_DELETE} \
            --insert-count ${POINTS_TO_INSERT} \
            --num-partitions ${NUM_PARTITION} \
            --size-t-bytes ${DATA_SIZE} \
            --dtype ${TYPE} \
            --gpu



    # --- 4. 현재 이터레이션의 Ground Truth 계산 ---
    GT_PATH_NEW="${GT_PREFIX}${PATH_MODIFIER_NEW}"
    
    run_with_log "compute_groundtruth_iter_${iteration}" \
        ${UTILS_DIR}/compute_groundtruth --data_type ${TYPE} \
                                        --dist_fn ${DIST_FN} \
                                        --base_file ${INDEX_PATH_NEW}.data \
                                        --query_file ${QUERY_PATH} \
                                        --K 100 --gt_file ${GT_PATH_NEW}

    # --- 5. 현재 이터레이션의 인덱스 검색 및 트레이스 ---
    DIR_MODIFIER_NEW="${PATH_MODIFIER_NEW#-}"
    RESULT_DIR_ITER="${RESULT_DIR}/${DIR_MODIFIER_NEW}"
    mkdir -p "${RESULT_DIR_ITER}"
    RESULT_PATH_PREFIX_NEW="${RESULT_DIR_ITER}/res"
    
    run_with_log "search_memory_index_with_trace_iter_${iteration}" \
        ${BUILD_DIR}/search_memory_index_with_trace  --data_type ${TYPE} \
                                                    --dist_fn ${DIST_FN} \
                                                    --index_path_prefix ${INDEX_PATH_NEW} \
                                                    --query_file ${QUERY_PATH} \
                                                    --gt_file ${GT_PATH_NEW} \
                                                    -K 10 -L 10 20 30 40 50 100 \
                                                    --result_path ${RESULT_PATH_PREFIX_NEW} \
                                                    --partition_file ${PARTITION_PATH_NEW}
    
    # --- 6. 현재 이터레이션의 결과 분석 ---
    ANALYSIS_DIR_ITER="${ANALYSIS_DIR}/${DIR_MODIFIER_NEW}"
    mkdir -p "${ANALYSIS_DIR_ITER}"

    run_with_log "visualize_points" \
        conda run -n rapids python -u ${TOOLS_DIR}/visualize_partitions.py \
                                            --partition_file ${PARTITION_PATH_NEW} \
                                            --output_image ${ANALYSIS_DIR_ITER}/points_analysis \
                                            --size_t_bytes ${DATA_SIZE}
    
    run_with_log "analyze_edges_iter_${iteration}" \
        conda run -n rapids python -u ${TOOLS_DIR}/analyze_edges.py --graph_file ${INDEX_PATH_NEW} \
                                             --partition_file ${PARTITION_PATH_NEW} \
                                             --output_prefix ${ANALYSIS_DIR_ITER}/edge_analysis \
                                             --num_partitions ${NUM_PARTITION} \
                                             --size_t_bytes ${DATA_SIZE}


    TRACE_FILE_NEW="${RESULT_DIR_ITER}/res_trace.csv"
    
    run_with_log "analyze_trace_iter_${iteration}" \
        conda run -n rapids python -u ${TOOLS_DIR}/analyze_trace.py --trace_file ${TRACE_FILE_NEW} \
                                             --output_prefix ${ANALYSIS_DIR_ITER}/trace_analysis\

    run_with_log "analyze_degree_${iteration}" \
        conda run -n rapids python -u ${TOOLS_DIR}/analyze_degree.py \
                                            --graph_file ${INDEX_PATH_NEW} \
                                            --output_prefix ${ANALYSIS_DIR_ITER}/degree_analysis \
                                            --size_t_bytes ${DATA_SIZE}


    # --- 7. 다음 반복을 위해 상태 변수 업데이트 ---
    INDEX_PATH_OLD=${INDEX_PATH_NEW}
    CURRENT_START_POINT=${NEXT_START_POINT}
    CURRENT_INDEX_SIZE=${NEXT_INDEX_SIZE}
done

# ================================================================
# Phase 3 : Re-partition and Analyze
# ================================================================
echo "---------- PHASE 3 : Re-partition and Analyze ----------"
# 변수를 Phase 1 이후 상태로 리셋하여, 각 데이터 윈도우를 순회
iteration=0
CURRENT_START_POINT=${START_POINT}
CURRENT_INDEX_SIZE=${INITIAL_INDEX_SIZE}

while [ $iteration -le ${MAX_ITERATIONS} ]; do
    iteration=$((iteration + 1))
    echo "---------- Iteration ${iteration} (Re-partition) ----------"
    
    PATH_MODIFIER_CURRENT="-from-${CURRENT_START_POINT}-to-$((CURRENT_START_POINT + CURRENT_INDEX_SIZE))"
    INDEX_PATH_CURRENT="${INDEX_PREFIX}${PATH_MODIFIER_CURRENT}"
    
    INCREMENTAL_PARTITION_PATH="${INDEX_PATH_CURRENT}.partition${NUM_PARTITION}"
    PARTITION_PATH_REPARTITION="${INDEX_PATH_CURRENT}-repartition.partition${NUM_PARTITION}"
    CENTROID_PATH_REPARTITION="${INDEX_PATH_CURRENT}-repartition.centroids${NUM_PARTITION}"
    
    run_with_log "repartition_index_${iteration}" \
        conda run -n rapids python -u ${TOOLS_DIR}/repartition_index.py \
            --data_file ${INDEX_PATH_CURRENT}.data \
            --existing_partition_file ${INCREMENTAL_PARTITION_PATH} \
            --output_partition_file ${PARTITION_PATH_REPARTITION} \
            -k ${NUM_PARTITION} \
            --output_centroid_file ${CENTROID_PATH_REPARTITION} \
            --dtype ${TYPE} --size_t_bytes ${DATA_SIZE} --dimension ${DIMENSION} --gpu

    # --- (이하 GT 계산, Search, Analysis 부분은 이전과 동일하게 유지) ---
    GT_PATH_CURRENT="${GT_PREFIX}${PATH_MODIFIER_CURRENT}"
    run_with_log "compute_groundtruth_repartition_iter_${iteration}" \
        ${UTILS_DIR}/compute_groundtruth --data_type ${TYPE} \
                                        --dist_fn ${DIST_FN} \
                                        --base_file ${INDEX_PATH_CURRENT}.data \
                                        --query_file ${QUERY_PATH} \
                                        --K 100 --gt_file ${GT_PATH_CURRENT}

    DIR_MODIFIER_CURRENT="${PATH_MODIFIER_CURRENT#-}"
    RESULT_DIR_REPARTITION="${RESULT_DIR}/${DIR_MODIFIER_CURRENT}-repartitioned"
    mkdir -p "${RESULT_DIR_REPARTITION}"
    RESULT_PATH_PREFIX_REPARTITION="${RESULT_DIR_REPARTITION}/res"

    run_with_log "search_with_trace_repartition_iter_${iteration}" \
        ${BUILD_DIR}/search_memory_index_with_trace  --data_type ${TYPE} \
                                                    --dist_fn ${DIST_FN} \
                                                    --index_path_prefix ${INDEX_PATH_CURRENT} \
                                                    --query_file ${QUERY_PATH} \
                                                    --gt_file ${GT_PATH_CURRENT} \
                                                    -K 10 -L 10 20 30 40 50 100 \
                                                    --result_path ${RESULT_PATH_PREFIX_REPARTITION} \
                                                    --partition_file ${PARTITION_PATH_REPARTITION}

    ANALYSIS_DIR_REPARTITION="${ANALYSIS_DIR}/${DIR_MODIFIER_CURRENT}-repartitioned"
    mkdir -p "${ANALYSIS_DIR_REPARTITION}"
    
    run_with_log "analyze_all_repartitioned_iter_${iteration}" \
        conda run -n rapids python -u ${TOOLS_DIR}/visualize_partitions.py --partition_file ${PARTITION_PATH_REPARTITION} --output_image ${ANALYSIS_DIR_REPARTITION}/points_analysis.png --size_t_bytes ${DATA_SIZE} && \
        conda run -n rapids python -u ${TOOLS_DIR}/analyze_edges.py --graph_file ${INDEX_PATH_CURRENT} --partition_file ${PARTITION_PATH_REPARTITION} --output_prefix ${ANALYSIS_DIR_REPARTITION}/edge_analysis --num_partitions ${NUM_PARTITION} --size_t_bytes ${DATA_SIZE} && \
        conda run -n rapids python -u ${TOOLS_DIR}/analyze_trace.py --trace_file "${RESULT_PATH_PREFIX_REPARTITION}_trace.csv" --output_prefix ${ANALYSIS_DIR_REPARTITION}/trace_analysis && \
        conda run -n rapids python -u ${TOOLS_DIR}/analyze_degree.py --graph_file ${INDEX_PATH_CURRENT} --output_prefix ${ANALYSIS_DIR_REPARTITION}/degree_analysis --size_t_bytes ${DATA_SIZE}
    
    # --- 다음 상태로 넘어가기 위한 변수 업데이트 ---
    CURRENT_START_POINT=$((CURRENT_START_POINT + POINTS_TO_DELETE))
    CURRENT_INDEX_SIZE=$((CURRENT_INDEX_SIZE + POINTS_TO_INSERT - POINTS_TO_DELETE))
done



# ================================================================
# Phase 4 : Re-build and Analyze
# ================================================================
echo "---------- PHASE 4 : Re-build and Analyze ----------"
# 변수를 Phase 1 이후 상태로 리셋하여, 각 데이터 윈도우를 순회
iteration=0
CURRENT_START_POINT=${START_POINT}
CURRENT_INDEX_SIZE=${INITIAL_INDEX_SIZE}

while [ $iteration -le ${MAX_ITERATIONS} ]; do # iter 0, 1, 2 세 번 반복
    echo "---------- Re-building and Analyzing State of Iteration ${iteration} ----------"

    # --- 1. 현재 분석할 대상 상태의 경로 및 변수 정의 ---
    PATH_MODIFIER_CURRENT="-from-${CURRENT_START_POINT}-to-$((CURRENT_START_POINT + CURRENT_INDEX_SIZE))"
    
    # Re-build된 인덱스를 저장할 새로운 경로 정의
    INDEX_PATH_REBUILT="${INDEX_PREFIX}${PATH_MODIFIER_CURRENT}-rebuilt"

    # --- 2. 해당 상태의 데이터로 그래프 Re-build 수행 ---
    run_with_log "Re-build_iter_${iteration}" \
        ${BUILD_DIR}/build_memory_index_offset --data_type ${TYPE} \
                                              --dist_fn ${DIST_FN} \
                                              --index_path_prefix ${INDEX_PATH_REBUILT} \
                                              --data_path ${DATA_PATH} \
                                              --points_to_skip ${CURRENT_START_POINT} \
                                              --beginning_index_size ${CURRENT_INDEX_SIZE} \
                                              -R 64 -L 100


    # --- 3. Re-build된 인덱스에 대해 파티션 새로 생성 ---
    PARTITION_PATH_REBUILT="${INDEX_PATH_REBUILT}.partition${NUM_PARTITION}"
    CENTROID_PATH_REBUILT="${INDEX_PATH_REBUILT}.centroids${NUM_PARTITION}"
    
    run_with_log "Create_Partition_on_Rebuilt_iter_${iteration}" \
        conda run -n rapids python -u ${TOOLS_DIR}/create_partitions.py \
            --data_file ${INDEX_PATH_REBUILT}.data \
            --partition_file ${PARTITION_PATH_REBUILT} \
            --num_partitions ${NUM_PARTITION} \
            --centroid_file  ${CENTROID_PATH_REBUILT} \
            --dtype ${TYPE} --size_t_bytes ${DATA_SIZE} --dimension ${DIMENSION} --gpu

    # --- 4. Re-build된 인덱스에 대한 GT, Search, Analysis 수행 ---
    GT_PATH_CURRENT="${GT_PREFIX}${PATH_MODIFIER_CURRENT}"
    DIR_MODIFIER_CURRENT="${PATH_MODIFIER_CURRENT#-}"
    
    GT_PATH_CURRENT="${GT_PREFIX}${PATH_MODIFIER_CURRENT}"
    run_with_log "compute_groundtruth_repartition_iter_${iteration}" \
        ${UTILS_DIR}/compute_groundtruth --data_type ${TYPE} \
                                        --dist_fn ${DIST_FN} \
                                        --base_file ${INDEX_PATH_REBUILT}.data \
                                        --query_file ${QUERY_PATH} \
                                        --K 100 --gt_file ${GT_PATH_CURRENT}


    # 분석 결과를 저장할 고유한 하위 디렉터리 생성
    ANALYSIS_DIR_REBUILT="${ANALYSIS_DIR}/${DIR_MODIFIER_CURRENT}-rebuilt"
    RESULT_DIR_REBUILT="${RESULT_DIR}/${DIR_MODIFIER_CURRENT}-rebuilt"
    mkdir -p "${ANALYSIS_DIR_REBUILT}" "${RESULT_DIR_REBUILT}"
    
    RESULT_PATH_PREFIX_REBUILT="${RESULT_DIR_REBUILT}/res"

    run_with_log "search_with_trace_rebuilt_iter_${iteration}" \
        ${BUILD_DIR}/search_memory_index_with_trace  --data_type ${TYPE} \
                                                    --dist_fn ${DIST_FN} \
                                                    --index_path_prefix ${INDEX_PATH_REBUILT} \
                                                    --query_file ${QUERY_PATH} \
                                                    --gt_file ${GT_PATH_CURRENT} \
                                                    -K 10 -L 10 20 30 40 50 100 \
                                                    --result_path ${RESULT_PATH_PREFIX_REBUILT} \
                                                    --partition_file ${PARTITION_PATH_REBUILT}

    TRACE_FILE_REBUILT="${RESULT_PATH_PREFIX_REBUILT}_trace.csv"
    run_with_log "analyze_all_rebuilt_iter_${iteration}" \
        conda run -n rapids python -u ${TOOLS_DIR}/analyze_trace.py --trace_file ${TRACE_FILE_REBUILT} --output_prefix ${ANALYSIS_DIR_REBUILT}/trace_analysis && \
        conda run -n rapids python -u ${TOOLS_DIR}/visualize_partitions.py --partition_file ${PARTITION_PATH_REBUILT} --output_image ${ANALYSIS_DIR_REBUILT}/points_analysis.png --size_t_bytes ${DATA_SIZE} && \
        conda run -n rapids python -u ${TOOLS_DIR}/analyze_edges.py --graph_file ${INDEX_PATH_REBUILT} --partition_file ${PARTITION_PATH_REBUILT} --output_prefix ${ANALYSIS_DIR_REBUILT}/edge_analysis --num_partitions ${NUM_PARTITION} --size_t_bytes ${DATA_SIZE} && \
        conda run -n rapids python -u ${TOOLS_DIR}/analyze_degree.py --graph_file ${INDEX_PATH_REBUILT} --output_prefix ${ANALYSIS_DIR_REBUILT}/degree_analysis --size_t_bytes ${DATA_SIZE}

    # --- 5. 다음 상태로 넘어가기 위한 변수 업데이트 ---
    CURRENT_START_POINT=$((CURRENT_START_POINT + POINTS_TO_DELETE))
    CURRENT_INDEX_SIZE=$((CURRENT_INDEX_SIZE + POINTS_TO_INSERT - POINTS_TO_DELETE))
    iteration=$((iteration + 1))
done

# -------------------------------------------------------------

echo "모든 시뮬레이션 완료."
