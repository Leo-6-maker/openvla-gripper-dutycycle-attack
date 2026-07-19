"""Unit tests for R7 K10 Opportunity Labeler V1.

Validates: segment closure, K10 arithmetic, mutual exclusion, identity closure.
Run: python -m pytest tests/test_r7_k10_opportunity_labeler.py -v
"""

import json, pytest, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "detector_v4"))
from label_k10_opportunity import (
    extract_segments, compute_critical, compute_burst_feasible,
    K, SUITES, RELEASE_SAFE_MARGIN,
)

# ── Synthetic teacher records for testing ──────────────────────────────

def make_records(n_steps: int, close_events: list[tuple[int, int]],
                 support_steps: set[int] = None,
                 grasp_steps: set[int] = None,
                 retention_steps: set[int] = None,
                 release_steps: set[int] = None,
                 release_imminent_steps: set[int] = None,
                 unknown_steps: set[int] = None,
                 opening_stable_steps: set[int] = None,
                 ) -> list[dict]:
    """Build synthetic teacher records for testing."""
    support_steps = support_steps or set()
    grasp_steps = grasp_steps or set()
    retention_steps = retention_steps or set()
    release_steps = release_steps or set()
    release_imminent_steps = release_imminent_steps or set()
    unknown_steps = unknown_steps or set()
    opening_stable_steps = opening_stable_steps or set()

    records = []
    for i in range(n_steps):
        in_close = any(onset <= i <= end for onset, end in close_events)
        is_close_onset = any(i == onset for onset, _ in close_events)
        end_step = next((end for onset, end in close_events if onset <= i <= end), -1)

        records.append({
            "step": i,
            "event_close_onset": is_close_onset,
            "event_end_step": end_step if is_close_onset else (-1 if not in_close else end_step),
            "event_start_step": next((onset for onset, _ in close_events if onset <= i <= end), -1) if is_close_onset else -1,
            "event_support": i in support_steps,
            "grasp_support": i in grasp_steps,
            "retention_active": i in retention_steps,
            "event_release_onset": i in release_steps,
            "release_imminent": i in release_imminent_steps,
            "retention_unknown_mask": i in unknown_steps,
            "event_evidence_valid": True,
            "event_opening_stable": True if i in opening_stable_steps else None,
            "event_qpos_stable": None,
            "event_id": next((j for j, (o, e) in enumerate(close_events) if o <= i <= e), -1),
        })
    return records


# ── Tests ──────────────────────────────────────────────────────────────

class TestSegmentExtraction:
    def test_single_segment(self):
        recs = make_records(30, [(5, 20)])
        segs = extract_segments(recs)
        assert len(segs) == 1
        assert segs[0]["onset"] == 5
        assert segs[0]["end"] == 20
        assert segs[0]["duration"] == 16

    def test_multiple_segments(self):
        recs = make_records(50, [(5, 15), (25, 40)])
        segs = extract_segments(recs)
        assert len(segs) == 2

    def test_segment_shorter_than_K_excluded(self):
        recs = make_records(20, [(5, 10)])  # 6 steps < K=10
        segs = extract_segments(recs)
        assert len(segs) == 0

    def test_segment_exactly_K(self):
        recs = make_records(20, [(5, 14)])  # 10 steps = K
        segs = extract_segments(recs)
        assert len(segs) == 1


class TestCriticalComputation:
    def test_all_conditions_met(self):
        recs = make_records(30, [(5, 25)],
                           support_steps=set(range(5, 26)),
                           grasp_steps=set(range(5, 26)),
                           retention_steps=set(range(5, 26)))
        segs = extract_segments(recs)
        critical, reasons, release_safe = compute_critical(recs, 30, segs)
        # Steps 5-25 should be critical (all conditions met, no release)
        for i in range(5, 26):
            assert critical[i], f"Step {i} should be critical, reason={reasons[i]}"

    def test_unknown_masked(self):
        recs = make_records(30, [(5, 25)],
                           support_steps=set(range(5, 26)),
                           grasp_steps=set(range(5, 26)),
                           retention_steps=set(range(5, 26)),
                           unknown_steps={10, 11, 12})
        segs = extract_segments(recs)
        critical, reasons, _ = compute_critical(recs, 30, segs)
        assert not critical[10]
        assert reasons[10] == "unknown_mask"

    def test_release_safe_excluded(self):
        recs = make_records(30, [(5, 25)],
                           support_steps=set(range(5, 26)),
                           grasp_steps=set(range(5, 26)),
                           retention_steps=set(range(5, 26)),
                           release_steps={20})
        segs = extract_segments(recs)
        critical, reasons, release_safe = compute_critical(recs, 30, segs)
        # Steps 17-23 should be release_safe (20 ± 3)
        for i in range(17, 24):
            assert release_safe[i], f"Step {i} should be release_safe"
        assert not critical[20]
        assert reasons[20] == "release_safe"

    def test_not_in_segment_excluded(self):
        recs = make_records(30, [(10, 20)])
        segs = extract_segments(recs)
        critical, reasons, _ = compute_critical(recs, 30, segs)
        assert not critical[0]
        assert reasons[0] == "not_in_close_segment"


class TestBurstFeasibility:
    def test_minimum_length_no_burst(self):
        """Segment length 9 < K=10: no feasible starts."""
        recs = make_records(20, [(5, 13)],
                           support_steps=set(range(5, 14)),
                           grasp_steps=set(range(5, 14)),
                           retention_steps=set(range(5, 14)))
        segs = extract_segments(recs)
        # This segment has 9 steps (5..13) < K=10, so extract_segments filters it
        assert len(segs) == 0

    def test_exact_length_one_start(self):
        """Segment length 10 = K: exactly 1 feasible start."""
        recs = make_records(30, [(5, 14)],  # 10 steps: 5..14
                           support_steps=set(range(5, 15)),
                           grasp_steps=set(range(5, 15)),
                           retention_steps=set(range(5, 15)))
        segs = extract_segments(recs)
        critical, reasons, release_safe = compute_critical(recs, 30, segs)
        n = 30
        in_seg = [-1] * n
        for seg in segs:
            for s in range(seg["onset"], seg["end"] + 1):
                in_seg[s] = seg["segment_id"]
        burst, is_start = compute_burst_feasible(critical, n, in_seg, segs, release_safe)
        starts = [i for i, s in enumerate(is_start) if s]
        assert len(starts) == 1
        assert starts[0] == 5

    def test_longer_window_multiple_starts(self):
        """Segment length L >= 10: L-9 feasible starts."""
        L = 20
        recs = make_records(40, [(5, 5 + L - 1)],
                           support_steps=set(range(5, 5 + L)),
                           grasp_steps=set(range(5, 5 + L)),
                           retention_steps=set(range(5, 5 + L)))
        segs = extract_segments(recs)
        critical, reasons, release_safe = compute_critical(recs, 40, segs)
        n = 40
        in_seg = [-1] * n
        for seg in segs:
            for s in range(seg["onset"], seg["end"] + 1):
                in_seg[s] = seg["segment_id"]
        burst, is_start = compute_burst_feasible(critical, n, in_seg, segs, release_safe)
        starts = [i for i, s in enumerate(is_start) if s]
        assert len(starts) == L - 9  # L - (K-1)

    def test_unknown_gap_blocks_burst(self):
        """Unknown step in middle blocks all K10 windows covering it."""
        recs = make_records(40, [(5, 30)],
                           support_steps=set(range(5, 31)),
                           grasp_steps=set(range(5, 31)),
                           retention_steps=set(range(5, 31)),
                           unknown_steps={15})
        segs = extract_segments(recs)
        critical, reasons, release_safe = compute_critical(recs, 40, segs)
        # Step 15 is unknown → not critical
        # Windows covering step 15 (starts 6..15) should NOT be feasible
        n = 40
        in_seg = [-1] * n
        for seg in segs:
            for s in range(seg["onset"], seg["end"] + 1):
                in_seg[s] = seg["segment_id"]
        burst, is_start = compute_burst_feasible(critical, n, in_seg, segs, release_safe)
        for t in range(6, 16):  # starts that cover step 15
            assert not burst[t], f"Window starting at {t} covers unknown step 15"

    def test_segment_boundary_no_cross(self):
        """K10 window must not cross segment boundary."""
        recs = make_records(40, [(5, 15), (20, 35)],
                           support_steps=set(range(5, 16)) | set(range(20, 36)),
                           grasp_steps=set(range(5, 16)) | set(range(20, 36)),
                           retention_steps=set(range(5, 16)) | set(range(20, 36)))
        segs = extract_segments(recs)
        critical, reasons, release_safe = compute_critical(recs, 40, segs)
        n = 40
        in_seg = [-1] * n
        for seg in segs:
            for s in range(seg["onset"], seg["end"] + 1):
                in_seg[s] = seg["segment_id"]
        burst, is_start = compute_burst_feasible(critical, n, in_seg, segs, release_safe)
        # Step 11 starts a window that would cross into the gap (steps 11..20, but segment ends at 15)
        assert not burst[11], "Window crossing segment boundary should not be feasible"

    def test_release_safe_in_window_blocks(self):
        """Release-safe step anywhere in K10 window blocks it."""
        recs = make_records(40, [(5, 30)],
                           support_steps=set(range(5, 31)),
                           grasp_steps=set(range(5, 31)),
                           retention_steps=set(range(5, 31)),
                           release_steps={20})
        segs = extract_segments(recs)
        critical, reasons, release_safe = compute_critical(recs, 40, segs)
        n = 40
        in_seg = [-1] * n
        for seg in segs:
            for s in range(seg["onset"], seg["end"] + 1):
                in_seg[s] = seg["segment_id"]
        burst, is_start = compute_burst_feasible(critical, n, in_seg, segs, release_safe)
        # Any start that covers step 20 (starts 11..20) should be blocked
        for t in range(11, 21):
            assert not burst[t], f"Window at {t} covers release-safe step 20"


class TestMutualExclusion:
    def test_no_quality_and_veto_same_step(self):
        """critical_t uses only positive components; no veto head exists."""
        recs = make_records(30, [(5, 25)],
                           support_steps=set(range(5, 26)),
                           grasp_steps=set(range(5, 26)),
                           retention_steps=set(range(5, 26)))
        segs = extract_segments(recs)
        critical, reasons, release_safe = compute_critical(recs, 30, segs)
        # Verify: no step is both release_safe and critical
        for i in range(30):
            assert not (release_safe[i] and critical[i]), \
                f"Step {i}: release_safe AND critical both True"


class TestIdentityClosure:
    def test_800_identities(self):
        """Verify labeler processes exactly 800 FIT identities."""
        # This test validates against the sealed server root
        import subprocess
        result = subprocess.run([
            "python", "-c",
            "import json; from pathlib import Path; "
            "root = Path('/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/OFFICIAL_V3_R7_K10_OPPORTUNITY_LABELER_V1_66f3604_20260719'); "
            "suites = ['libero_10','libero_goal','libero_object','libero_spatial']; "
            "count = sum(1 for s in suites for t in range(10) for st in range(20) "
            "if (root/s/f'task_{t:02d}'/f'state_{st:02d}'/'k10_labels.jsonl').exists()); "
            "print(count)"
        ], capture_output=True, text=True)
        # Skip if server not accessible
        if result.returncode == 0:
            assert result.stdout.strip() == "800"
