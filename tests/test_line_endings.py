"""Manifest-hashed files must reach a checkout byte-for-byte.

``pipeline_run_manifest.json`` records the SHA-256 of the exact schedule and
predictions bytes, and ``pregame_snapshots`` verifies them without any
normalization. With ``core.autocrlf=true`` (the Git for Windows default) Git
used to rewrite those files' LF line endings to CRLF on checkout, so a
correct checkout failed verification. ``.gitattributes`` now marks exactly
those two files ``-text``.

These tests use throwaway Git repositories with ``core.autocrlf=true`` set
explicitly, so they behave the same on Linux CI and on Windows. They never
touch the real ``data_files``.
"""

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

import pregame_snapshots as ps
from test_pregame_snapshots import PROBS, ROWS, SIGNALS, _manifest, _write_inputs
from test_team_features import ROOT

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

HASHED = [f"data_files/{ps.SCHEDULE_NAME}", f"data_files/{ps.PREDICTIONS_NAME}"]
CONTROL = "data_files/notes.txt"   # an ordinary text file, still converted


def git(repo, *args, check=True):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com",
         "-c", "core.safecrlf=false", *args],
        cwd=repo, capture_output=True, check=check)


def _to_lf(path):
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def _make_upstream(tmp_path, monkeypatch, with_attributes=True):
    """A repo whose committed bytes are LF, like the files the nightly commits."""
    up = tmp_path / "upstream"
    data = up / "data_files"
    git(tmp_path, "init", "-q", str(up))
    git(up, "config", "core.autocrlf", "false")
    _write_inputs(data, ROWS, PROBS, SIGNALS)
    for name in (ps.SCHEDULE_NAME, ps.PREDICTIONS_NAME):
        _to_lf(data / name)            # pandas writes CRLF on Windows
    _manifest(data, monkeypatch=monkeypatch)   # hashes the LF bytes
    (data / "code.py").unlink()
    (up / CONTROL).write_bytes(b"line one\nline two\n")
    if with_attributes:
        shutil.copy(ROOT / ".gitattributes", up / ".gitattributes")
    git(up, "add", "-A")
    git(up, "commit", "-qm", "data")
    return up


def _clone(tmp_path, up, name="checkout"):
    dest = tmp_path / name
    git(tmp_path, "clone", "-q", "-c", "core.autocrlf=true", str(up), str(dest))
    return dest


def _blob(repo, path):
    return git(repo, "show", f"HEAD:{path}").stdout


class TestRepositoryAttributes:
    def test_hashed_files_are_binary_safe(self):
        out = git(ROOT, "check-attr", "text", "--", *HASHED).stdout.decode()
        assert out.splitlines() == [f"{p}: text: unset" for p in HASHED]

    def test_rules_are_narrow(self):
        # Only the hashed files are exempt; other data files keep normal handling.
        others = ["data_files/model_metrics.json", "data_files/pipeline_run_manifest.json",
                  "data_files/betting_recommendations_log.csv", "README.md"]
        out = git(ROOT, "check-attr", "text", "--", *others).stdout.decode()
        assert all(line.endswith(": text: unspecified") for line in out.splitlines())


class TestFreshCheckout:
    def test_hashed_files_keep_exact_bytes(self, tmp_path, monkeypatch):
        up = _make_upstream(tmp_path, monkeypatch)
        wc = _clone(tmp_path, up)
        for path in HASHED:
            assert (wc / path).read_bytes() == _blob(up, path)
            assert b"\r\n" not in (wc / path).read_bytes()
        # The fixture really exercises conversion: an ordinary file gets CRLF.
        assert (wc / CONTROL).read_bytes() == b"line one\r\nline two\r\n"

    def test_checkout_passes_verification(self, tmp_path, monkeypatch):
        wc = _clone(tmp_path, _make_upstream(tmp_path, monkeypatch))
        manifest, sched, preds = ps.load_verified_inputs(wc / "data_files")
        assert len(sched) == len(ROWS) and len(preds) == len(ROWS)

    def test_without_the_rules_verification_fails(self, tmp_path, monkeypatch):
        # The original problem, reproduced: autocrlf converts the files and the
        # unchanged, strict verification rejects them.
        wc = _clone(tmp_path, _make_upstream(tmp_path, monkeypatch, with_attributes=False))
        assert b"\r\n" in (wc / HASHED[1]).read_bytes()
        with pytest.raises(ps.ProvenanceError, match="does not match"):
            ps.load_verified_inputs(wc / "data_files")


class TestVerificationIsStrict:
    """Line endings are part of the hashed bytes; nothing is normalized."""

    @pytest.mark.parametrize("name", [ps.PREDICTIONS_NAME, ps.SCHEDULE_NAME])
    def test_crlf_copy_of_identical_content_is_rejected(self, tmp_path, monkeypatch, name):
        data = _make_upstream(tmp_path, monkeypatch) / "data_files"
        path = data / name
        lf = path.read_bytes()
        path.write_bytes(lf.replace(b"\n", b"\r\n"))
        with pytest.raises(ps.ProvenanceError, match="does not match"):
            ps.load_verified_inputs(data)
        path.write_bytes(lf)
        ps.load_verified_inputs(data)

    def test_read_bytes_once_hashes_raw_bytes(self, tmp_path):
        import hashlib
        p = tmp_path / "f.csv"
        p.write_bytes(b"a\tb\r\n1\t2\r\n")
        data, digest = ps.read_bytes_once(p)
        assert data == b"a\tb\r\n1\t2\r\n"
        assert digest == hashlib.sha256(b"a\tb\r\n1\t2\r\n").hexdigest()
        assert digest != hashlib.sha256(b"a\tb\n1\t2\n").hexdigest()


def _stale_checkout(tmp_path, monkeypatch):
    """A checkout cloned before the rules existed, which then pulls them."""
    up = _make_upstream(tmp_path, monkeypatch, with_attributes=False)
    wc = _clone(tmp_path, up)
    for path in HASHED:
        assert b"\r\n" in (wc / path).read_bytes()   # converted, as on Windows
    # Make it a long-lived checkout: file times well before the index, so
    # Git trusts its cached stat data instead of re-reading the files.
    old = time.time() - 7200
    for f in (wc / "data_files").iterdir():
        os.utime(f, (old, old))
    git(wc, "update-index", "--refresh")
    # Upstream adds the rules; the old checkout pulls them.
    shutil.copy(ROOT / ".gitattributes", up / ".gitattributes")
    git(up, "add", ".gitattributes")
    git(up, "commit", "-qm", "attributes")
    git(wc, "pull", "-q", "--ff-only")
    return up, wc


class TestRefreshExistingCheckout:
    """The procedure in docs/PREGAME_SNAPSHOTS.md for checkouts made before
    the rules existed."""

    @pytest.fixture
    def stale(self, tmp_path, monkeypatch):
        return _stale_checkout(tmp_path, monkeypatch)

    def test_pulling_the_rules_does_not_rewrite_files(self, stale):
        up, wc = stale
        for path in HASHED:
            assert (wc / path).read_bytes() != _blob(up, path)
            assert b"\r\n" in (wc / path).read_bytes()
        # Git's cached stat data still says they're clean, so neither
        # `git status` nor a plain `git checkout -- <file>` notices them.
        assert git(wc, "status", "--porcelain").stdout == b""
        git(wc, "checkout", "--", *HASHED)
        assert all(b"\r\n" in (wc / p).read_bytes() for p in HASHED)

    # The exit-code rules the documented scripts rely on.
    def test_exit_code_0_for_line_ending_only_difference(self, stale):
        _, wc = stale
        for path in HASHED:
            os.utime(wc / path)   # make Git re-read the file, not trust its cache
            assert git(wc, "diff", "--ignore-cr-at-eol", "--quiet", "--", path,
                       check=False).returncode == 0

    def test_exit_code_0_does_not_prove_a_line_ending_difference(self, tmp_path, monkeypatch):
        # A fresh checkout is already byte-identical and also gives 0.
        wc = _clone(tmp_path, _make_upstream(tmp_path, monkeypatch))
        for path in HASHED:
            assert git(wc, "diff", "--ignore-cr-at-eol", "--quiet", "--", path,
                       check=False).returncode == 0

    def test_exit_code_1_for_content_edit(self, stale):
        _, wc = stale
        path = wc / HASHED[1]
        path.write_bytes(path.read_bytes().replace(b"0.58", b"0.99"))
        assert git(wc, "diff", "--ignore-cr-at-eol", "--quiet", "--", HASHED[1],
                   check=False).returncode == 1

    def test_exit_code_above_1_on_failure(self, tmp_path):
        not_a_repo = tmp_path / "plain"
        not_a_repo.mkdir()
        rc = git(not_a_repo, "diff", "--ignore-cr-at-eol", "--quiet", "--", "x.csv",
                 check=False).returncode
        assert rc > 1


def _doc_snippet(language):
    """The refresh script for ``language`` from docs/PREGAME_SNAPSHOTS.md."""
    doc = (ROOT / "docs" / "PREGAME_SNAPSHOTS.md").read_text(encoding="utf-8")
    section = doc[doc.index("**Refreshing a checkout made before the rule.**"):]
    start = section.index(f"```{language}\n") + len(f"```{language}\n")
    return section[start:section.index("```", start)]


def _bash():
    if os.name == "nt":   # prefer Git Bash over the WSL launcher in System32
        git_exe = shutil.which("git")
        if git_exe:
            # git.exe lives in <Git>\cmd or <Git>\mingw64\bin; bash in <Git>\bin.
            for root in Path(git_exe).parents[:3]:
                cand = root / "bin" / "bash.exe"
                if cand.exists():
                    return [str(cand)]
        return None
    found = shutil.which("bash")
    return [found] if found else None


def _powershell():
    for name in ("pwsh", "powershell"):
        found = shutil.which(name)
        if found:
            return [found, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File"]
    return None


SHELLS = {"bash": (_bash, ".sh"), "powershell": (_powershell, ".ps1")}


@pytest.fixture(params=sorted(SHELLS))
def run_documented_refresh(request, tmp_path):
    """Run the documented refresh script, verbatim, in a checkout."""
    finder, suffix = SHELLS[request.param]
    command = finder()
    if command is None:
        pytest.skip(f"{request.param} not available")
    script = tmp_path / f"refresh{suffix}"
    script.write_text(_doc_snippet(request.param), encoding="utf-8", newline="\n")

    def run(cwd):
        out = subprocess.run([*command, str(script)], cwd=cwd, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        return out.stdout
    return run


class TestDocumentedRefreshScripts:
    """docs/PREGAME_SNAPSHOTS.md's PowerShell and Bash refresh scripts, run as written."""

    @pytest.fixture
    def stale(self, tmp_path, monkeypatch):
        return _stale_checkout(tmp_path, monkeypatch)

    def test_restores_unedited_files_exactly(self, stale, run_documented_refresh):
        up, wc = stale
        out = run_documented_refresh(wc)
        for path in HASHED:
            assert (wc / path).read_bytes() == _blob(up, path)
            assert f"{path} restored" in out
        assert git(wc, "status", "--porcelain").stdout == b""
        ps.load_verified_inputs(wc / "data_files")

    def test_never_touches_an_edited_file(self, stale, run_documented_refresh):
        up, wc = stale
        edited_path = wc / HASHED[1]
        edited = edited_path.read_bytes().replace(b"0.58", b"0.99")
        edited_path.write_bytes(edited)
        out = run_documented_refresh(wc)
        assert edited_path.read_bytes() == edited          # byte-for-byte untouched
        assert f"{HASHED[1]} has content changes - left untouched" in out
        assert (wc / HASHED[0]).read_bytes() == _blob(up, HASHED[0])   # the other one restored
        # Edited data doesn't match the manifest - with CRLF or after conversion.
        with pytest.raises(ps.ProvenanceError):
            ps.load_verified_inputs(wc / "data_files")
        _to_lf(edited_path)
        with pytest.raises(ps.ProvenanceError):
            ps.load_verified_inputs(wc / "data_files")

    def test_changes_nothing_when_git_fails(self, stale, run_documented_refresh, tmp_path):
        _, wc = stale
        plain = tmp_path / "not_a_repo"
        shutil.copytree(wc / "data_files", plain / "data_files")
        before = {p: (plain / p).read_bytes() for p in HASHED}
        out = run_documented_refresh(plain)
        assert {p: (plain / p).read_bytes() for p in HASHED} == before
        assert all(f"git diff failed for {p}" in out for p in HASHED)

    def test_failed_delete_is_reported_and_changes_nothing(self, stale, run_documented_refresh):
        up, wc = stale
        locked = wc / HASHED[1]
        before = locked.read_bytes()
        if os.name == "nt":
            # Python opens files without delete sharing, so another process
            # can't delete this one while it's open - like an editor or scanner.
            handle = open(locked, "rb")
            release = handle.close
        else:
            if os.geteuid() == 0:
                pytest.skip("root can delete from a read-only directory")
            data_dir = locked.parent
            data_dir.chmod(0o555)
            release = lambda: data_dir.chmod(0o755)   # noqa: E731
        try:
            out = run_documented_refresh(wc)
        finally:
            release()
        assert locked.read_bytes() == before
        assert f"could not delete {HASHED[1]} - nothing changed" in out
        assert f"{HASHED[1]} restored" not in out

    def test_already_exact_checkout_is_left_identical(self, tmp_path, monkeypatch,
                                                      run_documented_refresh):
        up = _make_upstream(tmp_path, monkeypatch)
        wc = _clone(tmp_path, up)
        run_documented_refresh(wc)
        for path in HASHED:
            assert (wc / path).read_bytes() == _blob(up, path)
