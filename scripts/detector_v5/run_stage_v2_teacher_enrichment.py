"""Production entry point for the read-only Stage V2 enrichment runner."""
from __future__ import annotations

try:
    from .stage_v2_teacher_enrichment import main
except ImportError:  # pragma: no cover - direct server execution.
    from stage_v2_teacher_enrichment import main


if __name__ == "__main__":
    raise SystemExit(main())
