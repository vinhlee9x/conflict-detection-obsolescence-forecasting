import hashlib

import pandas as pd
import pytest

from src.datasets import jira_setup


def test_os_codename_reads_os_release(tmp_path):
    os_release = tmp_path / "os-release"
    os_release.write_text('NAME="Ubuntu"\nVERSION_CODENAME=jammy\n')
    assert jira_setup.os_codename(os_release) == "jammy"


def test_apt_source_line_targets_codename_and_series():
    line = jira_setup.apt_source_line("noble", "8.0")
    assert "ubuntu noble/mongodb-org/8.0 multiverse" in line
    assert "mongodb-server-8.0.gpg" in line


def test_restore_command_streams_only_selected_collections(tmp_path):
    command = jira_setup.restore_command(tmp_path / "dump with space.zip", ["Hyperledger", "Spring"])
    assert command.startswith("set -o pipefail; unzip -p ")
    assert "'" + str(tmp_path / "dump with space.zip") + "'" in command
    assert "--nsInclude=JiraReposAnon.Hyperledger" in command
    assert "--nsInclude=JiraReposAnon.Spring" in command
    assert "--gzip --archive " in command


def test_md5sum_matches_hashlib(tmp_path):
    path = tmp_path / "blob"
    path.write_bytes(b"jira" * 1000)
    assert jira_setup.md5sum(path, chunk_size=7) == hashlib.md5(b"jira" * 1000).hexdigest()


def test_suggest_projects_prefers_label_links_within_size_window():
    overview = pd.DataFrame({
        "project": ["HUGE", "SMALL", "GOOD", "OK"],
        "issues": [50_000, 100, 3_000, 4_000],
        "links": [9_000, 50, 800, 900],
        "label_links": [2_000, 40, 300, 120],
    })
    assert jira_setup.suggest_projects(overview, count=2, max_issues=8000, min_issues=500) == ["GOOD", "OK"]


def test_suggest_projects_raises_when_nothing_fits():
    overview = pd.DataFrame({"project": ["A"], "issues": [10], "links": [1], "label_links": [1]})
    with pytest.raises(ValueError):
        jira_setup.suggest_projects(overview)
