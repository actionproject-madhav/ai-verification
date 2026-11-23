"""The README's headline numbers are recomputed from results/ on every run of the tests."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_the_headline_numbers_still_follow_from_the_result_files():
    done = subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "headline.py")], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout
