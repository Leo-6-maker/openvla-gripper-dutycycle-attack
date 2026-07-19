#!/usr/bin/env python3
"""R8 Action-Replay Rerender Canary: 20-identity trajectory-parity + render-determinism test.

Replays Official V3 recorded actions in LIBERO, captures rendered frames,
and verifies trajectory parity against original 25D/EEF/qpos data.
"""

from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys, time, uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import imageio.v2 as imageio

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from gripper_attack.b3_training_protocol import verify_sealed_directory, sha256_file

SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
CANARY_PER_SUITE = 5


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _seal_root(root: Path) -> str:
    exclude = {"SHA256SUMS", "SHA256SUMS.sha256"}
    files = sorted([f for f in root.rglob("*") if f.is_file() and f.name not in exclude],
                   key=lambda f: str(f.relative_to(root)))
    lines = []
    for fp in files:
        rel = str(fp.relative_to(root)).replace("\\", "/")
        lines.append(f"{hashlib.sha256(fp.read_bytes()).hexdigest()}  {rel}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sha = hashlib.sha256((root / "SHA256SUMS").read_bytes()).hexdigest()
    (root / "SHA256SUMS.sha256").write_text(f"{sha}  SHA256SUMS\n", encoding="utf-8")
    return sha


def load_benchmark(suite: str):
    from libero.libero.benchmark import get_benchmark
    return get_benchmark(suite)()


def make_env(bench, task_idx: int, gpu_id: int = 0):
    from libero.libero.envs import OffScreenRenderEnv
    bddl_path = bench.get_task_bddl_file_path(task_idx)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=256, camera_widths=256,
        render_gpu_device_id=gpu_id,
    )
    env.seed(0)
    return env


def replay_episode(identity: str, clean_root: Path, gpu_id: int = 0) -> dict[str, Any]:
    parts = identity.split("/")
    suite, task_str, state_str = parts[0], parts[1], parts[2]
    task_idx = int(task_str.replace("task_", ""))
    state_id = int(state_str.replace("state_", ""))

    ep_dir = clean_root / suite / task_str / state_str
    meta = json.loads((ep_dir / "episode_metadata.json").read_text(encoding="utf-8"))
    step_records = _jsonl(ep_dir / "step_records.jsonl")
    teacher_sidecar = _jsonl(ep_dir / "privileged_teacher_sidecar.jsonl")
    n_steps = len(step_records)

    if len(teacher_sidecar) != n_steps:
        return {"status": "FAIL", "reason": f"sidecar: {len(teacher_sidecar)} vs {n_steps}"}

    applied_actions = [np.array(r["applied_action_7d"], dtype=np.float64) for r in step_records]

    bench = load_benchmark(suite)
    env = make_env(bench, task_idx, gpu_id)

    # Reset and try to match initial state
    obs = env.reset()
    initial_sha = meta.get("initial_state_sha256", "")
    if initial_sha:
        try:
            from libero.libero.envs.bddl_utils import get_problem_info
            init_states = bench.get_task_init_states(task_idx)
            if init_states:
                env.sim.set_state(init_states[0])
                obs = env._get_observations()
        except Exception as e:
            pass  # initial state restoration is best-effort

    # Replay actions, capture frames, verify parity
    frame_hashes_pass1: list[str] = []
    parity_errors: list[dict] = []
    step_count_ok = True
    done = False

    for t in range(n_steps):
        if done:
            step_count_ok = False
            break
        act = applied_actions[t]
        obs, reward, done, info = env.step(act)

        # Capture rendered frame
        img = obs.get("agentview_image")
        if img is not None:
            img_rotated = np.rot90(img, 2)
            img_bytes = imageio.imwrite(imageio.RETURN_BYTES, img_rotated, format="png")
            frame_hashes_pass1.append(hashlib.sha256(img_bytes).hexdigest())

        # Verify parity against teacher_sidecar
        sidecar = teacher_sidecar[t]
        orig_eef = np.array(sidecar.get("robot0_eef_pos", [0.0, 0.0, 0.0]), dtype=np.float64)
        orig_qpos_raw = sidecar.get("robot0_gripper_qpos", 0.0)
        orig_qpos = float(orig_qpos_raw[0]) if isinstance(orig_qpos_raw, list) else float(orig_qpos_raw)
        eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
        gripper_qpos = obs.get("robot0_gripper_qpos", np.zeros(2))

        eef_err = float(np.max(np.abs(eef_pos.astype(np.float64) - orig_eef)))
        gq_err = abs(float(gripper_qpos[0]) - orig_qpos) if len(gripper_qpos) > 0 else 0.0

        if eef_err > 1e-5 or gq_err > 1e-5:
            parity_errors.append({"step": t, "eef_error": eef_err, "gripper_error": gq_err})
            if len(parity_errors) >= 20:
                break

    env.close()

    # Second pass: render determinism
    frame_hashes_pass2: list[str] = []
    env2 = make_env(bench, task_idx, gpu_id)
    env2.reset()
    try:
        init_states = bench.get_task_init_states(task_idx)
        if init_states and initial_sha:
            env2.sim.set_state(init_states[0])
            env2._get_observations()
    except Exception:
        pass

    done = False
    for t in range(min(n_steps, len(frame_hashes_pass1))):
        if done:
            break
        obs2, _, done, _ = env2.step(applied_actions[t])
        img2 = obs2.get("agentview_image")
        if img2 is not None:
            img2_rotated = np.rot90(img2, 2)
            img2_bytes = imageio.imwrite(imageio.RETURN_BYTES, img2_rotated, format="png")
            frame_hashes_pass2.append(hashlib.sha256(img2_bytes).hexdigest())
    env2.close()

    render_deterministic = len(frame_hashes_pass1) == len(frame_hashes_pass2) and \
        all(a == b for a, b in zip(frame_hashes_pass1, frame_hashes_pass2))

    parity_pass = len(parity_errors) == 0
    all_pass = parity_pass and render_deterministic and step_count_ok

    return {
        "identity": identity,
        "n_steps_recorded": n_steps,
        "n_steps_replayed": len(frame_hashes_pass1),
        "n_frames": len(frame_hashes_pass1),
        "parity_errors": parity_errors[:5],
        "parity_pass": parity_pass,
        "step_count_ok": step_count_ok,
        "render_deterministic": render_deterministic,
        "status": "PASS" if all_pass else "FAIL",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-root", type=Path, required=True)
    ap.add_argument("--fold-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--identities", type=str, default="")
    args = ap.parse_args()

    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"output exists: {out}")
    staging = out.with_name(f".{out.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)

    try:
        git_commit = _git_commit()
        print(f"=== R8 ACTION-REPLAY RERENDER CANARY ===\nGit: {git_commit}")

        from gripper_attack.b3_training_protocol import load_fit_fold_bundle
        fold = load_fit_fold_bundle(args.fold_root)
        fold0 = next(f for f in fold["folds"] if f["fold_id"] == 0)
        train_ids = sorted(fold0["train_identities"])

        if args.identities:
            selected = [i.strip() for i in args.identities.split(",")]
        else:
            selected = []
            suite_groups: dict[str, list[str]] = defaultdict(list)
            for i in train_ids:
                suite_groups[i.split("/")[0]].append(i)
            for suite in SUITES:
                selected.extend(suite_groups[suite][:CANARY_PER_SUITE])
        print(f"Selected {len(selected)} identities")

        results = []
        for identity in selected:
            print(f"Replaying: {identity} ...", end=" ", flush=True)
            t0 = time.time()
            result = replay_episode(identity, args.clean_root, args.gpu)
            elapsed = time.time() - t0
            icon = "PASS" if result["status"] == "PASS" else "FAIL"
            print(f"{icon} ({elapsed:.0f}s) parity={result['parity_pass']} render_det={result['render_deterministic']} steps={result['n_steps_recorded']}/{result['n_steps_replayed']}")
            results.append(result)

        n_pass = sum(1 for r in results if r["status"] == "PASS")
        all_pass = n_pass == len(results)
        print(f"\n=== CANARY: {'PASS' if all_pass else 'FAIL'} ({n_pass}/{len(results)}) ===")

        (staging / "REPLAY_RESULTS.json").write_text(json.dumps({
            "schema": "R8_ACTION_REPLAY_CANARY_V1",
            "git_commit": git_commit,
            "n_identities": len(selected), "n_pass": n_pass, "all_pass": all_pass,
            "results": results,
        }, indent=2) + "\n", encoding="utf-8")
        (staging / "MANIFEST.json").write_text(json.dumps({
            "schema": "R8_ACTION_REPLAY_CANARY_MANIFEST_V1", "canary_pass": all_pass,
        }, indent=2) + "\n", encoding="utf-8")
        (staging / "commands.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

        root_sha = _seal_root(staging)
        os.replace(staging, out)
        print(f"Root: {out}\nSHA256SUMS: {root_sha}")

    except Exception:
        import shutil
        if staging.exists(): shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
