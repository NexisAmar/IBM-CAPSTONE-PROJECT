"""
Run all unit tests with a single script.

Usage:
    python run_tests.py

Exit codes:
    0 - All tests passed
    1 - One or more tests failed
"""

import os
import sys
import unittest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))


def run_all_tests():
    """Discover and run all unit tests in the tests/ directory."""
    print("=" * 70)
    print("AAVAIL Revenue Prediction — Running All Unit Tests")
    print("=" * 70)
    print()

    # Discover tests
    loader = unittest.TestLoader()
    test_dir = os.path.join(os.path.dirname(__file__), "tests")
    suite = loader.discover(test_dir, pattern="test_*.py")

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print()
    print("=" * 70)
    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)
    passed = total - failures - errors - skipped

    print(f"Total:    {total}")
    print(f"Passed:   {passed}")
    print(f"Failed:   {failures}")
    print(f"Errors:   {errors}")
    print(f"Skipped:  {skipped}")
    print("=" * 70)

    if result.wasSuccessful():
        print("STATUS: ALL TESTS PASSED ✓")
        return 0
    else:
        print("STATUS: SOME TESTS FAILED ✗")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
