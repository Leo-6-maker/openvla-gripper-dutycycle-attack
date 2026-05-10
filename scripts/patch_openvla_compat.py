#!/usr/bin/env python3
"""Patch OpenVLA local model files for offline/transformers>=5 compatibility (idempotent)."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _safe_backup(path: Path, suffix: str = '.bak_for_tfm5') -> None:
    bak = path.with_name(path.name + suffix)
    if not bak.exists():
        shutil.copy2(path, bak)


def _patch_json(path: Path, fn):
    _safe_backup(path)
    obj = json.loads(path.read_text(encoding='utf-8'))
    changed = fn(obj)
    if changed:
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    return changed


def _patch_processing_py(path: Path) -> bool:
    txt = path.read_text(encoding='utf-8')
    # already compatible -> no-op
    if 'tokenization_utils_base' in txt:
        return False

    old = 'from transformers.tokenization_utils import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy\n'
    new = (
        'try:\n'
        '    from transformers.tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy\n'
        'except Exception:\n'
        '    from transformers.tokenization_utils import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy\n'
    )
    if old in txt:
        _safe_backup(path)
        txt = txt.replace(old, new)
        path.write_text(txt, encoding='utf-8')
        return True
    return False


def _patch_modeling_py(path: Path) -> bool:
    txt = path.read_text(encoding='utf-8')
    changed = False

    old_block = (
        '        if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}:\n'
        '            raise NotImplementedError(\n'
        '                "TIMM Version must be >= 0.9.10 and < 1.0.0 (breaking); please raise a GitHub Issue "\n'
        '                "if you urgently need support for latest TIMM versions."\n'
        '            )\n'
    )
    new_block = (
        '        if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}:\n'
        '            logger.warning(\n'
        '                f"Running with unvalidated timm version `{timm.__version__}`; OpenVLA upstream expected 0.9.x. "\n'
        '                f"Proceeding for inference compatibility."\n'
        '            )\n'
    )
    if old_block in txt:
        txt = txt.replace(old_block, new_block)
        changed = True

    if 'def tie_weights(self) -> None:' in txt:
        txt = txt.replace('def tie_weights(self) -> None:', 'def tie_weights(self, *args, **kwargs) -> None:')
        changed = True

    if changed:
        _safe_backup(path)
        path.write_text(txt, encoding='utf-8')
    return changed


def patch_openvla(base_model_dir: Path, goal_model_dir: Path) -> dict:
    changed = {'copied_files': [], 'patched_files': []}

    for name in ['configuration_prismatic.py', 'modeling_prismatic.py', 'processing_prismatic.py']:
        src = base_model_dir / name
        dst = goal_model_dir / name
        if not dst.exists():
            shutil.copy2(src, dst)
            changed['copied_files'].append(str(dst))

    def _cfg_patch(obj: dict) -> bool:
        am = obj.get('auto_map', {})
        before = dict(am)
        am['AutoConfig'] = 'configuration_prismatic.OpenVLAConfig'
        am['AutoModelForVision2Seq'] = 'modeling_prismatic.OpenVLAForActionPrediction'
        am['AutoModelForImageTextToText'] = 'modeling_prismatic.OpenVLAForActionPrediction'
        am['AutoProcessor'] = 'processing_prismatic.PrismaticProcessor'
        obj['auto_map'] = am
        return am != before

    for md in [base_model_dir, goal_model_dir]:
        p = md / 'config.json'
        if _patch_json(p, _cfg_patch):
            changed['patched_files'].append(str(p))

    def _pre_patch(obj: dict) -> bool:
        am = obj.get('auto_map', {})
        before = dict(am)
        am['AutoImageProcessor'] = 'processing_prismatic.PrismaticImageProcessor'
        am['AutoProcessor'] = 'processing_prismatic.PrismaticProcessor'
        obj['auto_map'] = am
        return am != before

    def _tok_patch(obj: dict) -> bool:
        am = obj.get('auto_map', {})
        before = dict(am)
        am['AutoProcessor'] = 'processing_prismatic.PrismaticProcessor'
        obj['auto_map'] = am
        return am != before

    gp = goal_model_dir / 'preprocessor_config.json'
    if gp.exists() and _patch_json(gp, _pre_patch):
        changed['patched_files'].append(str(gp))

    gt = goal_model_dir / 'tokenizer_config.json'
    if gt.exists() and _patch_json(gt, _tok_patch):
        changed['patched_files'].append(str(gt))

    for md in [base_model_dir, goal_model_dir]:
        pp = md / 'processing_prismatic.py'
        mp = md / 'modeling_prismatic.py'
        if pp.exists() and _patch_processing_py(pp):
            changed['patched_files'].append(str(pp))
        if mp.exists() and _patch_modeling_py(mp):
            changed['patched_files'].append(str(mp))

    return changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-model-dir', default='${OPENVLA_BASE_MODEL_DIR}')
    ap.add_argument('--goal-model-dir', default='${OPENVLA_BASE_MODEL_DIR}-finetuned-libero-goal')
    args = ap.parse_args()

    result = patch_openvla(Path(args.base_model_dir), Path(args.goal_model_dir))
    print('[ok] patch done')
    print('copied_files:', len(result['copied_files']))
    print('patched_files:', len(result['patched_files']))
    for p in result['copied_files'] + result['patched_files']:
        print(' -', p)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
