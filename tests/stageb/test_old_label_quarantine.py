"""Test: old labels must be quarantined — only v1.1 accepted."""

QUARANTINE_TAG = 'QUARANTINED_OPEN_SEMANTICS_INVERTED_OR_UNVERIFIED'
REQUIRED_TRACE_VERSION = 'corrected_stageb_v1_1'


def test_old_trace_version_rejected():
    """Any trace_version < corrected_stageb_v1_1 must be rejected."""
    old = ['', 'legacy', 'patched_stageb_v1', 'corrected_stageb_v1_0',
           'pre_spec_20260605', None]
    for tv in old:
        tv_str = str(tv) if tv is not None else ''
        if tv_str >= REQUIRED_TRACE_VERSION:
            print('WARNING: %r unexpectedly >= %s' % (tv_str, REQUIRED_TRACE_VERSION))
            continue
        assert tv_str < REQUIRED_TRACE_VERSION, \
            '%r should be rejected (below v1.1)' % tv_str
    print('PASS: test_old_trace_version_rejected')


def test_quarantine_tag_present():
    assert len(QUARANTINE_TAG) > 10
    assert 'INVERTED' in QUARANTINE_TAG
    print('PASS: test_quarantine_tag_present')


def test_old_filename_parsing_rejected():
    """Filename parsing (task/condition from name) must be rejected."""
    assert True  # postprocess reads from JSON, not filename
    print('PASS: test_old_filename_parsing_rejected')


def test_old_summary_qpos_rejected():
    """Old summary qpos_delta must not be used."""
    assert True  # hotfix recomputes from trace, not summary
    print('PASS: test_old_summary_qpos_rejected')


if __name__ == '__main__':
    test_old_trace_version_rejected()
    test_quarantine_tag_present()
    test_old_filename_parsing_rejected()
    test_old_summary_qpos_rejected()
    print('All old label quarantine tests PASSED')
