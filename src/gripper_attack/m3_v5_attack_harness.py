"""CPU-only artifact contract helpers for M3 arm-v5.2 fixed-frame runs.

These helpers intentionally do not launch OpenVLA, PGD, RAND, shuffled-gradient,
or LIBERO. They define the artifact and audit contract that future GPU runs must
produce after V5.1 freezes exact input frames.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


V5_2_CONDITIONS = (
    "TRUE_PGD21_SELECTIVE",
    "RAND21_SELECTIVE",
    "SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE",
)
V5_2_FORBIDDEN_SEEDS = {85, 86}
V5_2_FROZEN_SEED = 428198
V5_2_CANDIDATE_COUNT = 21
V5_2_TARGET_TOKEN = 31744
V5_2_ARM_MATCH_MIN = 5
V5_2_ARM_LEN = 6
V5_2_EPSILON = 6.0 / 255.0


@dataclass(frozen=True)
class CandidateAudit:
    condition: str
    candidate_index: int
    target_token: int
    arm_match_count: int
    official_margin: float
    linf: float
    score_invariant_status: str
    exact_7_token_count: int
    route_status: str

    @property
    def feasible(self) -> bool:
        return (
            self.target_token == V5_2_TARGET_TOKEN
            and self.arm_match_count >= V5_2_ARM_MATCH_MIN
            and self.exact_7_token_count == 7
            and self.score_invariant_status == "PASS"
            and self.route_status == "PASS"
            and self.linf <= V5_2_EPSILON + 1e-8
        )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp), str(path))
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    atomic_write_text(path, buf.getvalue())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def require_v5_2_seed(seed: int) -> None:
    if int(seed) in V5_2_FORBIDDEN_SEEDS:
        raise ValueError(f"legacy seed is forbidden for V5.2: {seed}")
    if int(seed) != V5_2_FROZEN_SEED:
        raise ValueError(f"V5.2 seed must be frozen seed {V5_2_FROZEN_SEED}, got {seed}")


def require_candidate_index(index: int) -> None:
    if int(index) < 0 or int(index) >= V5_2_CANDIDATE_COUNT:
        raise ValueError(f"candidate index must be in 0..20, got {index}")


def require_condition(condition: str) -> None:
    if condition not in V5_2_CONDITIONS:
        raise ValueError(f"unknown V5.2 condition: {condition}")


def safe_rel(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"unsafe relative path: {value!r}")
    return path


def verify_frozen_input_binding(row: Mapping[str, Any], *, capture_root: Path) -> None:
    required = (
        "raw_image_path",
        "raw_image_sha256",
        "processed_tensor_path",
        "processed_tensor_sha256",
        "prompt_token_ids",
        "prompt_token_ids_sha256",
        "commit",
    )
    for key in required:
        if row.get(key, "") in ("", None):
            raise ValueError(f"frozen input missing {key}")
    for rel_key, sha_key in (("raw_image_path", "raw_image_sha256"), ("processed_tensor_path", "processed_tensor_sha256")):
        path = capture_root / safe_rel(str(row[rel_key]))
        if not path.is_file():
            raise ValueError(f"frozen input artifact missing: {rel_key}")
        if sha256_file(path) != str(row[sha_key]):
            raise ValueError(f"frozen input sha mismatch: {rel_key}")
    if sha256_text(str(row["prompt_token_ids"])) != str(row["prompt_token_ids_sha256"]):
        raise ValueError("prompt token sha mismatch")


def condition_output_dir(root: Path, *, frame_id: str, condition: str) -> Path:
    require_condition(condition)
    if not frame_id or "/" in frame_id or "\\" in frame_id or ".." in frame_id:
        raise ValueError(f"unsafe frame id: {frame_id!r}")
    return root / "frames" / frame_id / condition


def candidate_path(root: Path, *, frame_id: str, condition: str, candidate_index: int) -> Path:
    require_candidate_index(candidate_index)
    return condition_output_dir(root, frame_id=frame_id, condition=condition) / f"candidate_{candidate_index:02d}.json"


def arm_match_count(clean_prefix: Iterable[int], actual_prefix: Iterable[int]) -> int:
    clean = [int(x) for x in clean_prefix]
    actual = [int(x) for x in actual_prefix]
    if len(clean) != V5_2_ARM_LEN or len(actual) != V5_2_ARM_LEN:
        raise ValueError("arm prefixes must both contain six tokens")
    return sum(1 for left, right in zip(clean, actual) if left == right)


def write_candidate_artifact(
    root: Path,
    *,
    frame_id: str,
    condition: str,
    candidate_index: int,
    payload: Mapping[str, Any],
) -> Path:
    require_condition(condition)
    require_candidate_index(candidate_index)
    path = candidate_path(root, frame_id=frame_id, condition=condition, candidate_index=candidate_index)
    data = dict(payload)
    data.update({"frame_id": frame_id, "condition": condition, "candidate_index": int(candidate_index)})
    write_json(path, data)
    return path


def candidate_from_payload(payload: Mapping[str, Any]) -> CandidateAudit:
    return CandidateAudit(
        condition=str(payload["condition"]),
        candidate_index=int(payload["candidate_index"]),
        target_token=int(payload["official_gripper_token"]),
        arm_match_count=int(payload["arm_match_count"]),
        official_margin=float(payload["official_target_margin"]),
        linf=float(payload["linf"]),
        score_invariant_status=str(payload["score_invariant_status"]),
        exact_7_token_count=len(payload.get("official_exact_7_tokens", [])),
        route_status=str(payload.get("route_status", "")),
    )


def load_condition_candidates(root: Path, *, frame_id: str, condition: str) -> list[CandidateAudit]:
    directory = condition_output_dir(root, frame_id=frame_id, condition=condition)
    paths = sorted(directory.glob("candidate_*.json"))
    audits = [candidate_from_payload(json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    indices = [item.candidate_index for item in audits]
    if indices != list(range(V5_2_CANDIDATE_COUNT)):
        raise ValueError(f"{condition} candidate indices must be exactly 0..20, got {indices}")
    if any(item.condition != condition for item in audits):
        raise ValueError(f"{condition} candidate artifact cross-contamination detected")
    return audits


def select_best_feasible(candidates: Iterable[CandidateAudit]) -> CandidateAudit | None:
    feasible = [item for item in candidates if item.feasible]
    if not feasible:
        return None
    feasible.sort(key=lambda item: (-item.official_margin, item.linf, item.candidate_index))
    return feasible[0]


def audit_frame_group(root: Path, *, frame_ids: Iterable[str], seed: int) -> dict[str, Any]:
    require_v5_2_seed(seed)
    frame_results: list[dict[str, Any]] = []
    for frame_id in frame_ids:
        selected: dict[str, CandidateAudit | None] = {}
        counts: dict[str, int] = {}
        for condition in V5_2_CONDITIONS:
            candidates = load_condition_candidates(root, frame_id=frame_id, condition=condition)
            counts[condition] = len([item for item in candidates if item.feasible])
            selected[condition] = select_best_feasible(candidates)
        true = selected["TRUE_PGD21_SELECTIVE"]
        rand = selected["RAND21_SELECTIVE"]
        shuffled = selected["SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE"]
        full_pass = (
            true is not None
            and rand is not None
            and shuffled is not None
            and true.official_margin > rand.official_margin
            and true.official_margin > shuffled.official_margin
        )
        frame_results.append(
            {
                "frame_id": frame_id,
                "frame_full_selective_pass": full_pass,
                "true_selected_margin": "" if true is None else true.official_margin,
                "rand_selected_margin": "" if rand is None else rand.official_margin,
                "shuffled_selected_margin": "" if shuffled is None else shuffled.official_margin,
                "true_feasible_count": counts["TRUE_PGD21_SELECTIVE"],
                "rand_feasible_count": counts["RAND21_SELECTIVE"],
                "shuffled_feasible_count": counts["SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE"],
            }
        )
    return {
        "audit_status": "PASS",
        "seed": int(seed),
        "frame_count": len(frame_results),
        "frame_full_selective_pass_count": sum(1 for row in frame_results if row["frame_full_selective_pass"]),
        "frames": frame_results,
    }
