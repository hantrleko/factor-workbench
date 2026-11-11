from pathlib import Path

from scripts.check_clean_runs import dirty_run_files


def test_dirty_marker_is_rejected(tmp_path: Path):
    release = tmp_path / "data" / "releases" / "sample"
    release.mkdir(parents=True)
    (release / "run.json").write_text('{"source_commit": "abc123-dirty"}', encoding="utf-8")
    hits = dirty_run_files(tmp_path)
    assert hits == [release / "run.json"]


def test_clean_commit_is_accepted(tmp_path: Path):
    release = tmp_path / "data" / "releases" / "sample"
    release.mkdir(parents=True)
    (release / "run.json").write_text('{"source_commit": "abc123"}', encoding="utf-8")
    assert dirty_run_files(tmp_path) == []
