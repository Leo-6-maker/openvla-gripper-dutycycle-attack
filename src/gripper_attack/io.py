from __future__ import annotations
import csv, hashlib, json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str) -> None:
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path: str, obj: Any) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: str, rows: list[dict]) -> None:
    ensure_dir(str(Path(path).parent))
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    cols = sorted(set().union(*(r.keys() for r in rows)))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def read_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_jsonable(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def make_run_id(config: dict) -> str:
    task = config.get("task_id", "task")
    trig = config.get("trigger_name", config.get("trigger", "trigger"))
    rho = float(config.get("rho", 0.0))
    seed = int(config.get("seed", 0))
    return f"v4_{task}_{trig}_rho{rho:g}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sha256_jsonable(config)[:8]}"
