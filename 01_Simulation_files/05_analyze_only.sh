#!/bin/bash
if [ -z "$BASH_VERSION" ]; then
    exec bash "$0" "$@"
fi

# ================================================================
# 스크립트 사용법 확인 및 경로 설정
# ================================================================
if [ -z "$1" ]; then
    echo "오류: 분석할 시뮬레이션 디렉터리 경로를 입력해주세요."
    echo "사용법: $0 <SIM_DIR_PATH>"
    exit 1
fi

# 최상위 디렉터리로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# 입력받은 시뮬레이션 디렉터리
SIM_DIR="$1"
echo "분석을 시작할 디렉터리: ${SIM_DIR}"

# ================================================================
# 기존 시뮬레이션 파라미터 (파일 경로 재구성을 위해 필요)
# ================================================================
POINTS_TO_DELETE=25000
POINTS_TO_INSERT=40000
START_POINT=0
INITIAL_INDEX_SIZE=400000
NUM_PARTITION=8
DIMENSION=128
MAX_ITERATIONS=10

DATA_SIZE='4'


# ================================================================
# Path and Directory Setup
# ================================================================
INDEX_DIR="${SIM_DIR}/01_Index_files"
RESULT_DIR="${SIM_DIR}/05_Simulation_results"
ANALYSIS_DIR="${SIM_DIR}/06_Simulation_analysis/" # 분석 결과는 동일한 디렉터리에 덮어쓰거나 새로 생성
LOG_DIR="${SIM_DIR}/10_Log" # 로그는 동일한 디렉터리에 저장

INDEX_PREFIX="${INDEX_DIR}/index"
TOOLS_DIR="../00_Tools"

# 분석 결과 저장 디렉터리가 없다면 생성
mkdir -p ${ANALYSIS_DIR} ${LOG_DIR}

# ================================================================
# Helper Functions (Logging)
# ================================================================
LOG_STEP_COUNTER=0
run_with_log() {
    local step_name="$1"
    local step_prefix=$(printf "%02d" $LOG_STEP_COUNTER)
    local log_file="${LOG_DIR}/${step_prefix}_${step_name}_analysis_only.log"
    LOG_STEP_COUNTER=$((LOG_STEP_COUNTER + 1))
    shift
    
    echo -e "\n=== ${step_prefix} $step_name 분석 시작 - $(date) ==="
    "$@" 2>&1 | tee "$log_file"
    echo "=== ${step_prefix} $step_name 분석 완료 ==="
    echo ""
}

# ================================================================
# 스크립트 실행 시작
# ================================================================
CURRENT_START_POINT="${START_POINT}";
CURRENT_INDEX_SIZE=${INITIAL_INDEX_SIZE}

# ================================================================
# Phase 1 : Initial Data Analysis
# ================================================================
echo "---------- PHASE 1 : Initial Data Analysis ----------"
PATH_MODIFIER="-from-${CURRENT_START_POINT}-to-$((CURRENT_START_POINT + CURRENT_INDEX_SIZE))"
INDEX_PATH="${INDEX_PREFIX}${PATH_MODIFIER}"
PARTITION_PATH="${INDEX_PATH}.partition${NUM_PARTITION}"
DIR_MODIFIER="${PATH_MODIFIER#-}"
RESULT_DIR_INITIAL="${RESULT_DIR}/${DIR_MODIFIER}"
ANALYSIS_DIR_INIT="${ANALYSIS_DIR}/${DIR_MODIFIER}"
mkdir -p "${ANALYSIS_DIR_INIT}"

if [ -f "${PARTITION_PATH}" ]; then
    run_with_log "analyze_initial_partitions" \
        conda run -n rapids python -u ${TOOLS_DIR}/visualize_partitions.py \
            --partition_file ${PARTITION_PATH} \
            --output_image ${ANALYSIS_DIR_INIT}/points_analysis.png \
            --size_t_bytes ${DATA_SIZE}
fi

if [ -f "${INDEX_PATH}" ] && [ -f "${PARTITION_PATH}" ]; then
    run_with_log "analyze_initial_edges" \
        conda run -n rapids python -u ${TOOLS_DIR}/analyze_edges.py \
            --graph_file ${INDEX_PATH} \
            --partition_file ${PARTITION_PATH} \
            --output_prefix ${ANALYSIS_DIR_INIT}/edge_analysis \
            --num_partitions ${NUM_PARTITION} \
            --size_t_bytes ${DATA_SIZE}
fi

if [ -f "${RESULT_DIR_INITIAL}/res_trace.csv" ]; then
    run_with_log "analyze_initial_trace" \
        conda run -n rapids python -u ${TOOLS_DIR}/analyze_trace.py \
            --trace_file ${RESULT_DIR_INITIAL}/res_trace.csv \
            --output_prefix ${ANALYSIS_DIR_INIT}/trace_analysis
fi

if [ -f "${INDEX_PATH}" ]; then
    run_with_log "analyze_initial_degree" \
        conda run -n rapids python -u ${TOOLS_DIR}/analyze_degree.py \
            --graph_file ${INDEX_PATH} \
            --output_prefix ${ANALYSIS_DIR_INIT}/degree_analysis \
            --size_t_bytes ${DATA_SIZE}
fi

# ================================================================
# Phase 2 : Iterative Update Data Analysis
# ================================================================
echo "---------- PHASE 2 : Iterative Update Data Analysis ----------"

iteration=0
while [ $iteration -lt ${MAX_ITERATIONS} ]; do
    iteration=$((iteration + 1))
    
    CURRENT_START_POINT=$((CURRENT_START_POINT + POINTS_TO_DELETE))
    CURRENT_INDEX_SIZE=$((CURRENT_INDEX_SIZE + POINTS_TO_INSERT - POINTS_TO_DELETE))
    
    PATH_MODIFIER_NEW="-from-${CURRENT_START_POINT}-to-$((CURRENT_START_POINT + CURRENT_INDEX_SIZE))"
    INDEX_PATH_NEW="${INDEX_PREFIX}${PATH_MODIFIER_NEW}"
    PARTITION_PATH_NEW="${INDEX_PATH_NEW}.partition${NUM_PARTITION}"
    
    DIR_MODIFIER_NEW="${PATH_MODIFIER_NEW#-}"
    RESULT_DIR_ITER="${RESULT_DIR}/${DIR_MODIFIER_NEW}"
    ANALYSIS_DIR_ITER="${ANALYSIS_DIR}/${DIR_MODIFIER_NEW}"
    mkdir -p "${ANALYSIS_DIR_ITER}"

    echo "--- Analyzing Iteration ${iteration} Data (Path: ${DIR_MODIFIER_NEW}) ---"

    if [ -f "${PARTITION_PATH_NEW}" ]; then
        run_with_log "analyze_iter_${iteration}_partitions" \
            conda run -n rapids python -u ${TOOLS_DIR}/visualize_partitions.py \
                --partition_file ${PARTITION_PATH_NEW} \
                --output_image ${ANALYSIS_DIR_ITER}/points_analysis.png \
                --size_t_bytes ${DATA_SIZE}
    fi

    if [ -f "${INDEX_PATH_NEW}" ] && [ -f "${PARTITION_PATH_NEW}" ]; then
        run_with_log "analyze_iter_${iteration}_edges" \
            conda run -n rapids python -u ${TOOLS_DIR}/analyze_edges.py \
                --graph_file ${INDEX_PATH_NEW} \
                --partition_file ${PARTITION_PATH_NEW} \
                --output_prefix ${ANALYSIS_DIR_ITER}/edge_analysis \
                --num_partitions ${NUM_PARTITION} \
                --size_t_bytes ${DATA_SIZE}
    fi

    if [ -f "${RESULT_DIR_ITER}/res_trace.csv" ]; then
        run_with_log "analyze_iter_${iteration}_trace" \
            conda run -n rapids python -u ${TOOLS_DIR}/analyze_trace.py \
                --trace_file ${RESULT_DIR_ITER}/res_trace.csv \
                --output_prefix ${ANALYSIS_DIR_ITER}/trace_analysis
    fi

    if [ -f "${INDEX_PATH_NEW}" ]; then
        run_with_log "analyze_iter_${iteration}_degree" \
            conda run -n rapids python -u ${TOOLS_DIR}/analyze_degree.py \
                --graph_file ${INDEX_PATH_NEW} \
                --output_prefix ${ANALYSIS_DIR_ITER}/degree_analysis \
                --size_t_bytes ${DATA_SIZE}
    fi
done

# ================================================================
#  Phase 3 : Re-partition Data Analysis
# ================================================================
echo "---------- PHASE 3 : Re-partition Data Analysis ----------"
# 변수를 Phase 1 이후 상태로 리셋
iteration=0
CURRENT_START_POINT=${START_POINT}
CURRENT_INDEX_SIZE=${INITIAL_INDEX_SIZE}

while [ $iteration -lt ${MAX_ITERATIONS} ]; do
    iteration=$((iteration + 1))

    # --- 분석할 파일 경로 구성 ---
    PATH_MODIFIER_CURRENT="-from-${CURRENT_START_POINT}-to-$((CURRENT_START_POINT + CURRENT_INDEX_SIZE))"
    INDEX_PATH_CURRENT="${INDEX_PREFIX}${PATH_MODIFIER_CURRENT}"
    
    # Re-partitioned 파일 경로
    PARTITION_PATH_REPARTITION="${INDEX_PATH_CURRENT}-repartition.partition${NUM_PARTITION}"
    
    DIR_MODIFIER_CURRENT="${PATH_MODIFIER_CURRENT#-}"
    ANALYSIS_DIR_REPARTITION="${ANALYSIS_DIR}/${DIR_MODIFIER_CURRENT}-repartitioned"
    RESULT_DIR_REPARTITION="${RESULT_DIR}/${DIR_MODIFIER_CURRENT}-repartitioned"
    mkdir -p "${ANALYSIS_DIR_REPARTITION}"

    echo "--- Analyzing Repartitioned State of Iteration ${iteration} (Path: ${DIR_MODIFIER_CURRENT}-repartitioned) ---"

    # --- 분석 실행 (필요한 파일이 모두 존재하는지 확인) ---
    TRACE_FILE_REPARTITION="${RESULT_DIR_REPARTITION}/res_trace.csv"
    if [ ! -f "${PARTITION_PATH_REPARTITION}" ] || [ ! -f "${TRACE_FILE_REPARTITION}" ]; then
        echo "경고: Repartitioned 데이터 파일(${PARTITION_PATH_REPARTITION} 또는 ${TRACE_FILE_REPARTITION})을 찾을 수 없어 이번 이터레이션을 건너뜁니다."
    else
        run_with_log "analyze_repartitioned_${iteration}_all" \
            conda run -n rapids python -u ${TOOLS_DIR}/visualize_partitions.py --partition_file ${PARTITION_PATH_REPARTITION} --output_image ${ANALYSIS_DIR_REPARTITION}/points_analysis.png --size_t_bytes ${DATA_SIZE} && \
            conda run -n rapids python -u ${TOOLS_DIR}/analyze_edges.py --graph_file ${INDEX_PATH_CURRENT} --partition_file ${PARTITION_PATH_REPARTITION} --output_prefix ${ANALYSIS_DIR_REPARTITION}/edge_analysis --num_partitions ${NUM_PARTITION} --size_t_bytes ${DATA_SIZE} && \
            conda run -n rapids python -u ${TOOLS_DIR}/analyze_trace.py --trace_file "${TRACE_FILE_REPARTITION}" --output_prefix ${ANALYSIS_DIR_REPARTITION}/trace_analysis && \
            conda run -n rapids python -u ${TOOLS_DIR}/analyze_degree.py --graph_file ${INDEX_PATH_CURRENT} --output_prefix ${ANALYSIS_DIR_REPARTITION}/degree_analysis --size_t_bytes ${DATA_SIZE}
    fi

    # --- 다음 이터레이션을 위해 변수 업데이트 ---
    CURRENT_START_POINT=$((CURRENT_START_POINT + POINTS_TO_DELETE))
    CURRENT_INDEX_SIZE=$((CURRENT_INDEX_SIZE + POINTS_TO_INSERT - POINTS_TO_DELETE))
done



# ================================================================
# Phase 4 : Re-build Data Analysis
# ================================================================
echo "---------- PHASE 4 : Re-build Data Analysis ----------"
iteration=0
CURRENT_START_POINT=${START_POINT}
CURRENT_INDEX_SIZE=${INITIAL_INDEX_SIZE}

while [ $iteration -lt ${MAX_ITERATIONS} ]; do
    iteration=$((iteration + 1))

    PATH_MODIFIER_CURRENT="-from-${CURRENT_START_POINT}-to-$((CURRENT_START_POINT + CURRENT_INDEX_SIZE))"
    INDEX_PATH_REBUILT="${INDEX_PREFIX}${PATH_MODIFIER_CURRENT}-rebuilt"
    PARTITION_PATH_REBUILT="${INDEX_PATH_REBUILT}.partition${NUM_PARTITION}"
    
    DIR_MODIFIER_CURRENT="${PATH_MODIFIER_CURRENT#-}"
    ANALYSIS_DIR_REBUILT="${ANALYSIS_DIR}/${DIR_MODIFIER_CURRENT}-rebuilt"
    RESULT_DIR_REBUILT="${RESULT_DIR}/${DIR_MODIFIER_CURRENT}-rebuilt"
    mkdir -p "${ANALYSIS_DIR_REBUILT}"

    echo "--- Analyzing Rebuilt State of Iteration ${iteration} (Path: ${DIR_MODIFIER_CURRENT}-rebuilt) ---"

    if [ ! -f "${INDEX_PATH_REBUILT}" ]; then
        echo "경고: Rebuilt 데이터 파일(${INDEX_PATH_REBUILT})를 찾을 수 없어 이번 이터레이션을 건너뜁니다."
    else
        run_with_log "analyze_rebuilt_${iteration}_all" \
            conda run -n rapids python -u ${TOOLS_DIR}/visualize_partitions.py --partition_file ${PARTITION_PATH_REBUILT} --output_image ${ANALYSIS_DIR_REBUILT}/points_analysis.png --size_t_bytes ${DATA_SIZE} && \
            conda run -n rapids python -u ${TOOLS_DIR}/analyze_edges.py --graph_file ${INDEX_PATH_REBUILT} --partition_file ${PARTITION_PATH_REBUILT} --output_prefix ${ANALYSIS_DIR_REBUILT}/edge_analysis --num_partitions ${NUM_PARTITION} --size_t_bytes ${DATA_SIZE} && \
            conda run -n rapids python -u ${TOOLS_DIR}/analyze_trace.py --trace_file "${RESULT_DIR_REBUILT}/res_trace.csv" --output_prefix ${ANALYSIS_DIR_REBUILT}/trace_analysis && \
            conda run -n rapids python -u ${TOOLS_DIR}/analyze_degree.py --graph_file ${INDEX_PATH_REBUILT} --output_prefix ${ANALYSIS_DIR_REBUILT}/degree_analysis --size_t_bytes ${DATA_SIZE}
    fi

    CURRENT_START_POINT=$((CURRENT_START_POINT + POINTS_TO_DELETE))
    CURRENT_INDEX_SIZE=$((CURRENT_INDEX_SIZE + POINTS_TO_INSERT - POINTS_TO_DELETE))
done

echo "모든 분석 및 시각화가 완료되었습니다."