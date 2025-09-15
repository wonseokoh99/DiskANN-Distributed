#!/bin/bash
if [ -z "$BASH_VERSION" ]; then
    exec bash "$0" "$@"
fi

# ================================================================
# 스크립트 사용법 확인
# ================================================================
if [ -z "$1" ] || [[ "$1" == -* ]]; then
    echo "오류: 분석할 시뮬레이션 디렉터리 경로를 입력해주세요."
    echo "사용법: $0 <SIM_DIR_PATH> [--partitions] [--edges] [--trace] [--degree]"
    exit 1
fi

SIM_DIR="$1"
shift # 첫 번째 인자 (경로)를 제거하고 나머지 플래그만 남김

# ================================================================
# [수정] 실행할 분석을 선택하기 위한 플래그 파싱
# ================================================================
RUN_PARTITIONS=false
RUN_EDGES=false
RUN_TRACE=false
RUN_DEGREE=false
NO_FLAGS=true # 특정 플래그가 주어졌는지 확인

for arg in "$@"; do
    case $arg in
        --partitions)
            RUN_PARTITIONS=true
            NO_FLAGS=false
            shift
            ;;
        --edges)
            RUN_EDGES=true
            NO_FLAGS=false
            shift
            ;;
        --trace)
            RUN_TRACE=true
            NO_FLAGS=false
            shift
            ;;
        --degree)
            RUN_DEGREE=true
            NO_FLAGS=false
            shift
            ;;
    esac
done

# 아무 플래그도 없으면 모든 분석을 실행
if [ "$NO_FLAGS" = true ]; then
    echo "실행 플래그가 없습니다. 모든 분석을 실행합니다."
    RUN_PARTITIONS=true
    RUN_EDGES=true
    RUN_TRACE=true
    RUN_DEGREE=true
fi

# 최상위 디렉터리로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "분석을 시작할 디렉터리: ${SIM_DIR}"

# ================================================================
# 기존 시뮬레이션 파라미터
# ================================================================
POINTS_TO_DELETE=25000
POINTS_TO_INSERT=40000
START_POINT=0
INITIAL_INDEX_SIZE=400000
NUM_PARTITION=8
DIMENSION=128
MAX_ITERATIONS=8
DATA_SIZE='4'

# ================================================================
# Path and Directory Setup
# ================================================================
INDEX_DIR="${SIM_DIR}/01_Index_files"
RESULT_DIR="${SIM_DIR}/05_Simulation_results"
ANALYSIS_DIR="${SIM_DIR}/06_Simulation_analysis/"
LOG_DIR="${SIM_DIR}/10_Log"

INDEX_PREFIX="${INDEX_DIR}/index"
TOOLS_DIR="../00_Tools"

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

if [ "$RUN_PARTITIONS" = true ] && [ -f "${PARTITION_PATH}" ]; then
    run_with_log "analyze_initial_partitions" \
        conda run -n rapids python -u ${TOOLS_DIR}/visualize_partitions.py \
            --partition_file ${PARTITION_PATH} \
            --output_image ${ANALYSIS_DIR_INIT}/points_analysis.png \
            --size_t_bytes ${DATA_SIZE}
fi

if [ "$RUN_EDGES" = true ] && [ -f "${INDEX_PATH}" ] && [ -f "${PARTITION_PATH}" ]; then
    run_with_log "analyze_initial_edges" \
        conda run -n rapids python -u ${TOOLS_DIR}/analyze_edges.py \
            --graph_file ${INDEX_PATH} \
            --partition_file ${PARTITION_PATH} \
            --output_prefix ${ANALYSIS_DIR_INIT}/edge_analysis \
            --num_partitions ${NUM_PARTITION} \
            --size_t_bytes ${DATA_SIZE}
fi

if [ "$RUN_TRACE" = true ] && [ -f "${RESULT_DIR_INITIAL}/res_trace.csv" ]; then
    run_with_log "analyze_initial_trace" \
        conda run -n rapids python -u ${TOOLS_DIR}/analyze_trace.py \
            --trace_file ${RESULT_DIR_INITIAL}/res_trace.csv \
            --output_prefix ${ANALYSIS_DIR_INIT}/trace_analysis
fi

if [ "$RUN_DEGREE" = true ] && [ -f "${INDEX_PATH}" ]; then
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

    if [ "$RUN_PARTITIONS" = true ] && [ -f "${PARTITION_PATH_NEW}" ]; then
        run_with_log "analyze_iter_${iteration}_partitions" \
            conda run -n rapids python -u ${TOOLS_DIR}/visualize_partitions.py \
                --partition_file ${PARTITION_PATH_NEW} \
                --output_image ${ANALYSIS_DIR_ITER}/points_analysis.png \
                --size_t_bytes ${DATA_SIZE}
    fi

    if [ "$RUN_EDGES" = true ] && [ -f "${INDEX_PATH_NEW}" ] && [ -f "${PARTITION_PATH_NEW}" ]; then
        run_with_log "analyze_iter_${iteration}_edges" \
            conda run -n rapids python -u ${TOOLS_DIR}/analyze_edges.py \
                --graph_file ${INDEX_PATH_NEW} \
                --partition_file ${PARTITION_PATH_NEW} \
                --output_prefix ${ANALYSIS_DIR_ITER}/edge_analysis \
                --num_partitions ${NUM_PARTITION} \
                --size_t_bytes ${DATA_SIZE}
    fi

    if [ "$RUN_TRACE" = true ] && [ -f "${RESULT_DIR_ITER}/res_trace.csv" ]; then
        run_with_log "analyze_iter_${iteration}_trace" \
            conda run -n rapids python -u ${TOOLS_DIR}/analyze_trace.py \
                --trace_file ${RESULT_DIR_ITER}/res_trace.csv \
                --output_prefix ${ANALYSIS_DIR_ITER}/trace_analysis
    fi

    if [ "$RUN_DEGREE" = true ] && [ -f "${INDEX_PATH_NEW}" ]; then
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
iteration=0
CURRENT_START_POINT=${START_POINT}
CURRENT_INDEX_SIZE=${INITIAL_INDEX_SIZE}

while [ $iteration -lt ${MAX_ITERATIONS} ]; do
    iteration=$((iteration + 1))
    PATH_MODIFIER_CURRENT="-from-${CURRENT_START_POINT}-to-$((CURRENT_START_POINT + CURRENT_INDEX_SIZE))"
    INDEX_PATH_CURRENT="${INDEX_PREFIX}${PATH_MODIFIER_CURRENT}"
    PARTITION_PATH_REPARTITION="${INDEX_PATH_CURRENT}-repartition.partition${NUM_PARTITION}"
    DIR_MODIFIER_CURRENT="${PATH_MODIFIER_CURRENT#-}"
    ANALYSIS_DIR_REPARTITION="${ANALYSIS_DIR}/${DIR_MODIFIER_CURRENT}-repartitioned"
    RESULT_DIR_REPARTITION="${RESULT_DIR}/${DIR_MODIFIER_CURRENT}-repartitioned"
    mkdir -p "${ANALYSIS_DIR_REPARTITION}"

    echo "--- Analyzing Repartitioned State of Iteration ${iteration} (Path: ${DIR_MODIFIER_CURRENT}-repartitioned) ---"

    TRACE_FILE_REPARTITION="${RESULT_DIR_REPARTITION}/res_trace.csv"
    if [ ! -f "${PARTITION_PATH_REPARTITION}" ]; then
        echo "경고: Repartitioned 데이터 파일(${PARTITION_PATH_REPARTITION})을 찾을 수 없어 이번 이터레이션을 건너뜁니다."
    else
        if [ "$RUN_PARTITIONS" = true ]; then
            run_with_log "analyze_repartitioned_${iteration}_partitions" conda run -n rapids python -u ${TOOLS_DIR}/visualize_partitions.py --partition_file ${PARTITION_PATH_REPARTITION} --output_image ${ANALYSIS_DIR_REPARTITION}/points_analysis.png --size_t_bytes ${DATA_SIZE}
        fi
        if [ "$RUN_EDGES" = true ]; then
            run_with_log "analyze_repartitioned_${iteration}_edges" conda run -n rapids python -u ${TOOLS_DIR}/analyze_edges.py --graph_file ${INDEX_PATH_CURRENT} --partition_file ${PARTITION_PATH_REPARTITION} --output_prefix ${ANALYSIS_DIR_REPARTITION}/edge_analysis --num_partitions ${NUM_PARTITION} --size_t_bytes ${DATA_SIZE}
        fi
        if [ "$RUN_TRACE" = true ] && [ -f "${TRACE_FILE_REPARTITION}" ]; then
            run_with_log "analyze_repartitioned_${iteration}_trace" conda run -n rapids python -u ${TOOLS_DIR}/analyze_trace.py --trace_file "${TRACE_FILE_REPARTITION}" --output_prefix ${ANALYSIS_DIR_REPARTITION}/trace_analysis
        fi
        if [ "$RUN_DEGREE" = true ]; then
            run_with_log "analyze_repartitioned_${iteration}_degree" conda run -n rapids python -u ${TOOLS_DIR}/analyze_degree.py --graph_file ${INDEX_PATH_CURRENT} --output_prefix ${ANALYSIS_DIR_REPARTITION}/degree_analysis --size_t_bytes ${DATA_SIZE}
        fi
    fi
    
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
        if [ "$RUN_PARTITIONS" = true ]; then
            run_with_log "analyze_rebuilt_${iteration}_partitions" conda run -n rapids python -u ${TOOLS_DIR}/visualize_partitions.py --partition_file ${PARTITION_PATH_REBUILT} --output_image ${ANALYSIS_DIR_REBUILT}/points_analysis.png --size_t_bytes ${DATA_SIZE}
        fi
        if [ "$RUN_EDGES" = true ]; then
            run_with_log "analyze_rebuilt_${iteration}_edges" conda run -n rapids python -u ${TOOLS_DIR}/analyze_edges.py --graph_file ${INDEX_PATH_REBUILT} --partition_file ${PARTITION_PATH_REBUILT} --output_prefix ${ANALYSIS_DIR_REBUILT}/edge_analysis --num_partitions ${NUM_PARTITION} --size_t_bytes ${DATA_SIZE}
        fi
        if [ "$RUN_TRACE" = true ] && [ -f "${RESULT_DIR_REBUILT}/res_trace.csv" ]; then
            run_with_log "analyze_rebuilt_${iteration}_trace" conda run -n rapids python -u ${TOOLS_DIR}/analyze_trace.py --trace_file "${RESULT_DIR_REBUILT}/res_trace.csv" --output_prefix ${ANALYSIS_DIR_REBUILT}/trace_analysis
        fi
        if [ "$RUN_DEGREE" = true ]; then
            run_with_log "analyze_rebuilt_${iteration}_degree" conda run -n rapids python -u ${TOOLS_DIR}/analyze_degree.py --graph_file ${INDEX_PATH_REBUILT} --output_prefix ${ANALYSIS_DIR_REBUILT}/degree_analysis --size_t_bytes ${DATA_SIZE}
        fi
    fi

    CURRENT_START_POINT=$((CURRENT_START_POINT + POINTS_TO_DELETE))
    CURRENT_INDEX_SIZE=$((CURRENT_INDEX_SIZE + POINTS_TO_INSERT - POINTS_TO_DELETE))
done

echo "모든 분석 및 시각화가 완료되었습니다."