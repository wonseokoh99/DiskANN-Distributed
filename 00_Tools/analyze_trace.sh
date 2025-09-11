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

run_with_log "analyze_trace" \
    python analyze_trace.py --trace_file ../05_Simulation_results/sift/res_trace.csv --output_prefix ../06_Analysis/02_Trace
