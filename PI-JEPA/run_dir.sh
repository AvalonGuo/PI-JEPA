#!/usr/bin/env bash
# Shared run-dir convention for the PI-JEPA launchers (sourced, not executed):
#   <OUT_ROOT>/<task>-<alg>-<addition_info>_seed<seed>
# <addition_info> is the ONLY experiment tag, so a run dir never grows extra suffixes
# and scripts/eval_policy.sh resolves it without RUN_DIR=.
set -euo pipefail

ablation_run_dir() {   # out_root task alg addition_info seed
    printf '%s/%s-%s-%s_seed%s\n' "$1" "$2" "$3" "$4" "$5"
}

# Claim a run dir: refuse dirs this helper did not create (baseline, other experiments,
# older naming) so their numbers stay untouched; otherwise stamp the marker. Nothing is
# created under DRY_RUN=1.
ablation_claim_run_dir() {   # run_dir json_tag
    local run_dir=$1 tag=$2
    if [ -d "${run_dir}" ] && [ ! -f "${run_dir}/ablation.json" ]; then
        echo "[run_dir] refusing to write into a run dir this helper did not create:" >&2
        echo "[run_dir]   ${run_dir}" >&2
        echo "[run_dir] hint: <addition_info> names the dir -- pick a new one" >&2
        return 1
    fi
    if [ "${DRY_RUN:-0}" != "1" ]; then
        mkdir -p "${run_dir}"
        printf '%s\n' "${tag}" > "${run_dir}/ablation.json"
    fi
}
