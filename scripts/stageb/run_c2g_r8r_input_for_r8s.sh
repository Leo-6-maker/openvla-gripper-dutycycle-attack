#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_c2g_r8r_input_for_r8s.sh \
    --repo /path/to/clean/worktree \
    --expected-head <40-char-sha1> \
    --output-root /new/external/output/root

This is the R8S-bound R8R input launcher. It reruns the checked-in read-only
Clean2000 reuse audit on the exact R8S head, uses an explicit GitHub remote
without mutating origin, verifies both SHA ledgers, and stops.
EOF
}

REPO=""
EXPECTED_HEAD=""
OUTPUT_ROOT=""
REMOTE_URL="${R8S_REMOTE_URL:-https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack.git}"
BRANCH=assistant/c2g-r8s-semantic-replay-audit-20260711

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --expected-head) EXPECTED_HEAD="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$REPO" && -n "$EXPECTED_HEAD" && -n "$OUTPUT_ROOT" ]] || { usage >&2; exit 2; }
[[ "$EXPECTED_HEAD" =~ ^[0-9a-f]{40}$ ]] || { echo "expected head must be a full lowercase SHA1" >&2; exit 2; }

REPO="$(realpath "$REPO")"
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

R7_PLAN_ROOT=/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r7_corpus_plan_bb281d5_20260711
R7_PLAN="$R7_PLAN_ROOT/c2g_scientific_corpus_plan_report.json"
R7_REGISTRY="$R7_PLAN_ROOT/c2g_parent_registry.jsonl"
R7_SOURCE_ROOT=/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r7_source_inventory_bb281d5_20260711
R7_SOURCE_AUDIT="$R7_SOURCE_ROOT/c2g_r7_clean_source_inventory_audit.json"
R7_REUSABLE="$R7_SOURCE_ROOT/c2g_r7_reusable_clean_parents.jsonl"
EXPECTED_PLAN_SHA=29d28f05d85426be59eb451e096e6bea25e65aaab1d7905093282aba93f14b86
EXPECTED_REGISTRY_SHA=2b909797381752597f5a57d0cf37643f6a983f1a446f7642ef9cdd1b794580c0
EXPECTED_SOURCE_AUDIT_SHA=154466f35336602e48d6ce331c254281dbec1792c50c87f3d3a1ebfb86fc7216
EXPECTED_REUSABLE_SHA=6a4c16fc97b6fb2cdb411d6510de4e4d3c1c64b189894b49abc214370787d34a

RAW_ROOT=/mnt/sdc/dty_user/openvla_attack_evidence/c2f/clean2000_obs_clean_36712cc
MERGED_ROOT=/mnt/sdc/dty_user/openvla_attack_evidence/c2f/clean2000_merged_199af7b
REPLACEMENT_ROOT=/mnt/sdc/dty_user/openvla_attack_evidence/c2f/object500_v1.1_fd3e2db
PROVENANCE_1=/mnt/sdc/dty_user/openvla_attack_evidence/detector_dataset/clean2000_primary_source_root_resolution_af8217c.json
PROVENANCE_2=/mnt/sdc/dty_user/openvla_attack_evidence/detector_dataset/clean2000_source_provenance_audit_af8217c.json

for path in "$R7_PLAN" "$R7_REGISTRY" "$R7_SOURCE_AUDIT" "$R7_REUSABLE" "$PROVENANCE_1" "$PROVENANCE_2"; do
  [[ -f "$path" ]] || { echo "required file missing: $path" >&2; exit 1; }
done
for path in "$RAW_ROOT" "$MERGED_ROOT" "$REPLACEMENT_ROOT"; do
  [[ -d "$path" ]] || { echo "required source root missing: $path" >&2; exit 1; }
done
[[ "$(sha256sum "$R7_PLAN" | awk '{print $1}')" == "$EXPECTED_PLAN_SHA" ]] || { echo "R7 plan hash mismatch" >&2; exit 1; }
[[ "$(sha256sum "$R7_REGISTRY" | awk '{print $1}')" == "$EXPECTED_REGISTRY_SHA" ]] || { echo "R7 registry hash mismatch" >&2; exit 1; }
[[ "$(sha256sum "$R7_SOURCE_AUDIT" | awk '{print $1}')" == "$EXPECTED_SOURCE_AUDIT_SHA" ]] || { echo "R7 source audit hash mismatch" >&2; exit 1; }
[[ "$(sha256sum "$R7_REUSABLE" | awk '{print $1}')" == "$EXPECTED_REUSABLE_SHA" ]] || { echo "R7 reusable manifest hash mismatch" >&2; exit 1; }

SPEC_TMPDIR="$(mktemp -d /tmp/c2g_r8r_source_spec.XXXXXX)"
SPEC="$SPEC_TMPDIR/source_spec.json"
trap 'rm -rf "$SPEC_TMPDIR"' EXIT
PREDECESSOR_ARGS=()
while IFS= read -r root; do
  [[ -n "$root" ]] && PREDECESSOR_ARGS+=(--predecessor-root "$root")
done < <(
  find /mnt/sdc/dty_user/openvla_attack_evidence/c2g \
    -maxdepth 1 -type d \
    -name 'c2g_r8r_clean2000_reuse_audit_*_20260711*' \
    ! -path "$(realpath -m "$OUTPUT_ROOT")" \
    -print | sort
)

export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
python tools/multisuite_detector/build_c2g_r8r_clean2000_source_spec.py \
  --raw-root "$RAW_ROOT" \
  --merged-root "$MERGED_ROOT" \
  --replacement-root "$REPLACEMENT_ROOT" \
  --shared-evidence "$PROVENANCE_1" \
  --shared-evidence "$PROVENANCE_2" \
  "${PREDECESSOR_ARGS[@]}" \
  --output "$SPEC"

python tools/multisuite_detector/audit_c2g_r8r_clean2000_reuse.py \
  --registry "$R7_REGISTRY" \
  --plan-report "$R7_PLAN" \
  --expected-plan-report-sha256 "$EXPECTED_PLAN_SHA" \
  --source-audit-report "$R7_SOURCE_AUDIT" \
  --expected-source-audit-report-sha256 "$EXPECTED_SOURCE_AUDIT_SHA" \
  --reusable-manifest "$R7_REUSABLE" \
  --expected-reusable-manifest-sha256 "$EXPECTED_REUSABLE_SHA" \
  --source-spec "$SPEC" \
  --output-dir "$OUTPUT_ROOT" \
  --expected-git-commit "$EXPECTED_HEAD" \
  | tee "$OUTPUT_ROOT.console.json"

(
  cd "$OUTPUT_ROOT"
  sha256sum -c SHA256SUMS
  sha256sum -c SHA256SUMS.sha256
)
