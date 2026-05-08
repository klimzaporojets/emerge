"""Argparse / import smoke test for the 7 scripts ported from the dataset
construction pipeline.

Catches the most common breakage from a port: a renamed import, a
typo'd argparse default, an `argparse.FileType` pointing at a Snellius
path that no longer exists, a Python syntax error from an editor's
auto-format. None of these need the actual data on disk — they all
surface when the script is invoked with `--help`.

<2 s, CPU only, no network. The specific scripts are hand-listed (not
glob'd) so adding a new script doesn't silently bypass the test.
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

PORTED_SCRIPTS = [
    REPO / "scripts/dataset/find_garbage_clusters.py",
    REPO / "scripts/dataset/inspect_cluster_neighbors.py",
    REPO / "scripts/dataset/build_reinput_for_garbage.py",
    REPO / "scripts/dataset/merge_reinput_into_dataset.py",
    REPO / "scripts/stats/compute_405bv1_dataset_stats.py",
    REPO / "scripts/stats/figures_dataset_stats_v2_405bv1.py",
    REPO / "scripts/stats/figure_tkgu_distribution_v2_405bv1.py",
]


@pytest.mark.parametrize("script", PORTED_SCRIPTS, ids=lambda p: p.name)
def test_script_help_exits_zero(script):
    """`<script> --help` must import cleanly and exit 0 with usage on stdout."""
    assert script.exists(), f"missing: {script.relative_to(REPO)}"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, (
        f"{script.name} --help exited {result.returncode}\n"
        f"stderr:\n{result.stderr}"
    )
    # argparse always prints "usage: " somewhere when --help succeeds.
    assert "usage:" in result.stdout.lower(), (
        f"{script.name} --help produced no usage line; argparse may not be wired up"
    )
