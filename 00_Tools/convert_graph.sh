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
POINTS_TO_DELETE=25000
POINTS_TO_INSERT=40000
START_POINT=0
INITIAL_INDEX_SIZE=400000
MAX_ITERATIONS=10
DATA_OFFSET=0



# ================================================================
# Path and Directory Setup
# ================================================================
TIMESTAMP="20250907_063308"  # 또는 실제 타임스탬프
DATASET='sift'
SIM_NAME="04_build_partition_graph-smallest_partition"
SIM_PATH_MODIFIER="${SIM_NAME}-${DATASET}-${TIMESTAMP}"

# Main simulation directory
SIM_DIR="../03_Simulation_data/${SIM_PATH_MODIFIER}"
INDEX_DIR="${SIM_DIR}/01_Index_files"
ANALYSIS_DIR="${SIM_DIR}/06_Simulation_analysis"
LOG_DIR="${SIM_DIR}/10_Log"

INDEX_PREFIX="${INDEX_DIR}/index"

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

# Logging
LOG_STEP_COUNTER=0
run_with_log() {
    local step_name="$1"
    local step_prefix=$(printf "%02d" $LOG_STEP_COUNTER)
    local log_file="${LOG_DIR}/${step_prefix}_${step_name}_${TIMESTAMP}.log"
    LOG_STEP_COUNTER=$((LOG_STEP_COUNTER + 1))
    shift
    
    echo -e "\n=== ${step_prefix} $step_name 시작 - $(date) ==="
    
    # 명령어를 서브셸에서 실행하여 pipefail을 지역적으로 적용하고, 실실패 여부를 직접 확인합니다.
    # 'if ! (명령어)' 구문은 명령어가 실패했을 때(non-zero exit code) then 블록을 실행합니다.
    if ! (set -o pipefail; "$@" 2>&1 | tee "$log_file"); then
        # 실패 시, handle_failure 함수를 직접 호출합니다.
        handle_failure
    fi
    
    echo "=== ${step_prefix} $step_name 완료 ==="
    echo ""
}

# 출력 버퍼링 해제
export PYTHONUNBUFFERED=1

# ================================================================
# Phase 1: Convert Initial Graph (iteration 0)
# ================================================================
echo "---------- Phase 1: Converting Initial Graph ----------"
CURRENT_START_POINT=${START_POINT}
CURRENT_INDEX_SIZE=${INITIAL_INDEX_SIZE}

PATH_MODIFIER="-from-${CURRENT_START_POINT}-to-$((CURRENT_START_POINT + CURRENT_INDEX_SIZE))"
DIR_MODIFIER="${PATH_MODIFIER#-}"
INDEX_PATH="${INDEX_PREFIX}${PATH_MODIFIER}"
ANALYSIS_DIR_ITER="${ANALYSIS_DIR}/${DIR_MODIFIER}"

mkdir -p "${ANALYSIS_DIR_ITER}"

# Convert original graph
run_with_log "convert_graph_initial" \
    conda run -n rapids python -u convert_graph.py \
        --graph_file ${INDEX_PATH} \
        --data_offset ${DATA_OFFSET} \
        --output_prefix ${ANALYSIS_DIR_ITER}/res

# Convert rebuilt graph if exists
INDEX_PATH_REBUILT="${INDEX_PATH}-rebuilt"
if [ -f "${INDEX_PATH_REBUILT}" ]; then
    ANALYSIS_DIR_REBUILT="${ANALYSIS_DIR_ITER}-rebuilt"
    mkdir -p "${ANALYSIS_DIR_REBUILT}"
    
    run_with_log "convert_graph_initial_rebuilt" \
        conda run -n rapids python -u convert_graph.py \
            --graph_file ${INDEX_PATH_REBUILT} \
            --data_offset ${DATA_OFFSET} \
            --output_prefix ${ANALYSIS_DIR_REBUILT}/res
fi

# ================================================================
# Phase 2: Convert Iterative Graphs
# ================================================================
echo "---------- Phase 2: Converting Iterative Graphs ----------"
iteration=0
CURRENT_START_POINT=${START_POINT}
CURRENT_INDEX_SIZE=${INITIAL_INDEX_SIZE}

while [ $iteration -lt ${MAX_ITERATIONS} ]; do
    iteration=$((iteration + 1))
    echo "---------- Iteration ${iteration} Graph Conversion ----------"
    
    # --- 다음 단계의 변수들 계산 ---
    NEXT_START_POINT=$((CURRENT_START_POINT + POINTS_TO_DELETE))
    NEXT_INDEX_SIZE=$((CURRENT_INDEX_SIZE + POINTS_TO_INSERT - POINTS_TO_DELETE))
    
    PATH_MODIFIER_NEW="-from-${NEXT_START_POINT}-to-$((NEXT_START_POINT + NEXT_INDEX_SIZE))"
    DIR_MODIFIER_NEW="${PATH_MODIFIER_NEW#-}"
    INDEX_PATH_NEW="${INDEX_PREFIX}${PATH_MODIFIER_NEW}"
    ANALYSIS_DIR_ITER="${ANALYSIS_DIR}/${DIR_MODIFIER_NEW}"
    
    mkdir -p "${ANALYSIS_DIR_ITER}"
    
    # Convert original incremental graph
    if [ -f "${INDEX_PATH_NEW}" ]; then
        run_with_log "convert_graph_iter_${iteration}" \
            conda run -n rapids python -u convert_graph.py \
                --graph_file ${INDEX_PATH_NEW} \
                --data_offset ${DATA_OFFSET} \
                --output_prefix ${ANALYSIS_DIR_ITER}/res
    else
        echo "경고: 그래프 파일 ${INDEX_PATH_NEW}이 존재하지 않습니다."
    fi
    
    # Convert rebuilt graph if exists
    INDEX_PATH_REBUILT="${INDEX_PATH_NEW}-rebuilt"
    if [ -f "${INDEX_PATH_REBUILT}" ]; then
        ANALYSIS_DIR_REBUILT="${ANALYSIS_DIR_ITER}-rebuilt"
        mkdir -p "${ANALYSIS_DIR_REBUILT}"
        
        run_with_log "convert_graph_iter_${iteration}_rebuilt" \
            conda run -n rapids python -u convert_graph.py \
                --graph_file ${INDEX_PATH_REBUILT} \
                --data_offset ${DATA_OFFSET} \
                --output_prefix ${ANALYSIS_DIR_REBUILT}/res
    else
        echo "경고: Rebuilt 그래프 파일 ${INDEX_PATH_REBUILT}이 존재하지 않습니다."
    fi
    
    # --- 다음 반복을 위해 상태 변수 업데이트 ---
    CURRENT_START_POINT=${NEXT_START_POINT}
    CURRENT_INDEX_SIZE=${NEXT_INDEX_SIZE}
done

# ================================================================
# Phase 3: Graph Comparison Analysis
# ================================================================
echo "---------- Phase 3: Graph Comparison Analysis ----------"

# Create analysis output directory
COMPARISON_ANALYSIS_DIR="${ANALYSIS_DIR}/comparison_analysis"
mkdir -p "${COMPARISON_ANALYSIS_DIR}"

# Reset variables for comparison
iteration=0
CURRENT_START_POINT=${START_POINT}
CURRENT_INDEX_SIZE=${INITIAL_INDEX_SIZE}

while [ $iteration -le ${MAX_ITERATIONS} ]; do
    echo "---------- Comparing Iteration ${iteration} Graphs ----------"
    
    # Calculate current path
    PATH_MODIFIER_CURRENT="-from-${CURRENT_START_POINT}-to-$((CURRENT_START_POINT + CURRENT_INDEX_SIZE))"
    DIR_MODIFIER_CURRENT="${PATH_MODIFIER_CURRENT#-}"
    
    # Define CSV file paths
    GRAPH1_CSV="${ANALYSIS_DIR}/${DIR_MODIFIER_CURRENT}/res_nodes_detailed.csv"
    GRAPH2_CSV="${ANALYSIS_DIR}/${DIR_MODIFIER_CURRENT}-rebuilt/res_nodes_detailed.csv"
    
    # Check if both CSV files exist before comparison
    if [ -f "${GRAPH1_CSV}" ] && [ -f "${GRAPH2_CSV}" ]; then
        run_with_log "compare_graphs_iter_${iteration}" \
            conda run -n rapids python -u graph_comparison.py \
                --graph1_csv ${GRAPH1_CSV} \
                --graph2_csv ${GRAPH2_CSV} \
                --output_prefix ${COMPARISON_ANALYSIS_DIR}/comparison_iter_${iteration} \
                --save_detailed_csv
    else
        echo "경고: Iteration ${iteration}에서 비교할 CSV 파일이 없습니다."
        echo "  Graph1 CSV: ${GRAPH1_CSV} (exists: $([ -f "${GRAPH1_CSV}" ] && echo "YES" || echo "NO"))"
        echo "  Graph2 CSV: ${GRAPH2_CSV} (exists: $([ -f "${GRAPH2_CSV}" ] && echo "YES" || echo "NO"))"
    fi
    
    # --- 다음 상태로 넘어가기 위한 변수 업데이트 ---
    if [ $iteration -lt ${MAX_ITERATIONS} ]; then
        CURRENT_START_POINT=$((CURRENT_START_POINT + POINTS_TO_DELETE))
        CURRENT_INDEX_SIZE=$((CURRENT_INDEX_SIZE + POINTS_TO_INSERT - POINTS_TO_DELETE))
    fi
    
    iteration=$((iteration + 1))
done

echo "==========================================="
echo "모든 그래프 변환 및 비교 분석이 완료되었습니다!"
echo "결과는 다음 디렉터리에서 확인하실 수 있습니다:"
echo "  - CSV 파일: ${ANALYSIS_DIR}/"
echo "  - 비교 분석: ${COMPARISON_ANALYSIS_DIR}/"
echo "==========================================="