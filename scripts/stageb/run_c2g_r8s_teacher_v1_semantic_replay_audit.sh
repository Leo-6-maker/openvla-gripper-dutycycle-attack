#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_c2g_r8s_teacher_v1_semantic_replay_audit.sh \
    --repo /path/to/clean/worktree \
    --expected-head <40-char-sha1> \
    --r8r-root /path/to/final/r8r/root \
    --expected-r8r-report-sha256 <64-char-sha256> \
    --output-root /new/external/output/root \
    [--teacher-v1-source /path/to/authoritative/generator.py]

This launcher performs a read-only Teacher-v1 semantic-salvage and deterministic
replay-feasibility audit. It does not load models, import/create LIBERO environments,
execute replay, launch rollout, train, calibrate, or read attack outcomes.
EOF
}

REPO=""
EXPECTED_HEAD=""
R8R_ROOT=""
EXPECTED_R8R_REPORT_SHA256=""
OUTPUT_ROOT=""
TEACHER_V1_SOURCE=""
REMOTE_URL="${R8S_REMOTE_URL:-https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack.git}"
BRANCH=assistant/c2g-r8s-semantic-replay-audit-20260711

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --expected-head) EXPECTED_HEAD="$2"; shift 2 ;;
    --r8r-root) R8R_ROOT="$2"; shift 2 ;;
    --expected-r8r-report-sha256) EXPECTED_R8R_REPORT_SHA256="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --teacher-v1-source) TEACHER_V1_SOURCE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$REPO" && -n "$EXPECTED_HEAD" && -n "$R8R_ROOT" && -n "$EXPECTED_R8R_REPORT_SHA256" && -n "$OUTPUT_ROOT" ]] || {
  usage >&2
  exit 2
}
[[ "$EXPECTED_HEAD" =~ ^[0-9a-f]{40}$ ]] || { echo "expected head must be a full lowercase SHA1" >&2; exit 2; }
[[ "$EXPECTED_R8R_REPORT_SHA256" =~ ^[0-9a-f]{64}$ ]] || { echo "expected R8R report hash must be a full lowercase SHA256" >&2; exit 2; }

REPO="$(realpath "$REPO")"
R8R_ROOT="$(realpath "$R8R_ROOT")"
cd "$REPO"

LOCAL_HEAD="$(git rev-parse HEAD)"
WORKTREE_STATUS="$(git status --short)"
git fetch "$REMOTE_URL" "$BRANCH" --quiet
REMOTE_HEAD="$(git rev-parse FETCH_HEAD)"

[[ "$REMOTE_HEAD" == "$EXPECTED_HEAD" ]] || { echo "remote head mismatch: $REMOTE_HEAD != $EXPECTED_HEAD" >&2; exit 1; }
[[ "$LOCAL_HEAD" == "$EXPECTED_HEAD" ]] || { echo "local head mismatch: $LOCAL_HEAD != $EXPECTED_HEAD" >&2; exit 1; }
[[ -z "$WORKTREE_STATUS" ]] || { echo "worktree is not clean" >&2; printf '%s\n' "$WORKTREE_STATUS" >&2; exit 1; }
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "output root already exists: $OUTPUT_ROOT" >&2; exit 1; }
[[ ! -e "$OUTPUT_ROOT.console.json" ]] || { echo "console output already exists: $OUTPUT_ROOT.console.json" >&2; exit 1; }
[[ -d "$R8R_ROOT" ]] || { echo "R8R root missing: $R8R_ROOT" >&2; exit 1; }
[[ -f "$R8R_ROOT/clean2000_r7_reuse_audit_report.json" ]] || { echo "R8R report missing" >&2; exit 1; }
[[ "$(sha256sum "$R8R_ROOT/clean2000_r7_reuse_audit_report.json" | awk '{print $1}')" == "$EXPECTED_R8R_REPORT_SHA256" ]] || {
  echo "R8R report hash mismatch" >&2
  exit 1
}
if [[ -n "$TEACHER_V1_SOURCE" ]]; then
  TEACHER_V1_SOURCE="$(realpath "$TEACHER_V1_SOURCE")"
  [[ -f "$TEACHER_V1_SOURCE" ]] || { echo "Teacher-v1 source missing: $TEACHER_V1_SOURCE" >&2; exit 1; }
fi

export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
ARGS=(
  --repo "$REPO"
  --expected-git-commit "$EXPECTED_HEAD"
  --r8r-root "$R8R_ROOT"
  --expected-r8r-report-sha256 "$EXPECTED_R8R_REPORT_SHA256"
  --output-dir "$OUTPUT_ROOT"
)
if [[ -n "$TEACHER_V1_SOURCE" ]]; then
  ARGS+=(--teacher-v1-source "$TEACHER_V1_SOURCE")
fi

python tools/multisuite_detector/audit_c2g_r8s_teacher_v1_semantic_replay.py "${ARGS[@]}" \
  | tee "$OUTPUT_ROOT.console.json"

(
  cd "$OUTPUT_ROOT"
  sha256sum -c SHA256SUMS
  sha256sum -c SHA256SUMS.sha256
)

python - "$OUTPUT_ROOT" <<'PY'
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
report = json.loads((root / "r8s_semantic_replay_audit_report.json").read_text())
for key in (
    "r8s_head", "episode_count", "read_failure_count",
    "legacy_auxiliary_eligible_count", "strict_replay_candidate_count",
    "strict_replay_ready_count", "strict_replay_not_ready_count",
    "replay_canary_parent_count", "current_contract_uncovered_count",
    "exact_equivalent_mapping_count", "semantic_mapping_counts",
    "invariants", "final_decision", "next_recommended_stage",
):
    print(f"{key} = {json.dumps(report.get(key), sort_keys=True)}")
print("output_root =", root)
PY
