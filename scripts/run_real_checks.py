"""Supported verification: no oscillator fixtures or dummy-video generators.

Set DUVERA_REAL_DEMO to a finished real-footage project for delivery integration checks.
Legacy synthetic AV regressions are deliberately excluded from this runner.
"""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]


def main():
    loader = unittest.TestLoader()
    names = [
        'test_cinema_upgrade',
        'test_production_v2',
        'test_project_clock',
        'test_documentary_editor.SharedClock',
        'test_commentary.CommentaryPolicy',
        'test_pacing.SchedulerTests',
        'test_vocals.InstrumentalContracts.test_conflicting_prompt_is_sanitized_idempotently_and_bounded',
        'test_vocals.InstrumentalContracts.test_direct_provider_payload_keeps_flag_and_hard_suffix',
    ]
    suite = unittest.TestSuite(loader.loadTestsFromName(name) for name in names)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print('Real-media integration root:', os.environ.get('DUVERA_REAL_DEMO', 'not supplied; delivery integration skipped'))
    return not result.wasSuccessful()


if __name__ == '__main__':
    raise SystemExit(main())
