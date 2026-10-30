"""The "player-props-py312" CI job (.github/workflows/tests.yml) and its test
selection (tests/player_props_py312_suites.txt), checked offline."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUITES = ROOT / "tests" / "player_props_py312_suites.txt"
JOB = "player-props-py312"

# Areas the job must cover (one file at least each).
REQUIRED = [
    "tests/test_player_props_stage1.py", "tests/test_player_props_stage2.py",
    "tests/test_player_props_stage3.py", "tests/test_player_props_stage4.py",
    "tests/test_player_props_stage5a.py",
    "tests/test_player_props_rehearsal.py", "tests/test_player_props_rehearsal_real.py",
    "tests/test_player_props_rehearsal_early_failure.py",
    "tests/test_player_props_reference.py", "tests/test_player_props_position_group.py",
    "tests/test_player_props_reference_join.py", "tests/test_player_props_containment.py",
    "tests/test_pick6_wording.py", "tests/test_model_performance_week.py",
]


def suites():
    return [line.strip() for line in SUITES.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")]


@pytest.fixture(scope="module")
def workflow():
    import yaml
    return yaml.safe_load((ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8"))


class TestSelection:
    def test_every_entry_exists_once(self):
        listed = suites()
        assert len(listed) == len(set(listed))
        assert [p for p in listed if not (ROOT / p).is_file()] == []

    def test_every_player_props_test_file_is_listed(self):
        files = sorted(f"tests/{p.name}" for p in (ROOT / "tests").glob("test_player_props_*.py"))
        assert set(files) - set(suites()) == set()

    def test_required_areas_are_listed(self):
        assert set(REQUIRED) - set(suites()) == set()

    def test_selection_collects(self):
        """pytest collects every listed file without errors, with the job's socket guard loaded."""
        r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
                            "-p", "tests.no_network_plugin",
                            *[p for p in suites() if p != "tests/test_player_props_suites.py"]],
                           cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        assert "error" not in r.stdout.splitlines()[-1].lower()


class TestWorkflow:
    def test_existing_jobs_and_triggers_unchanged(self, workflow):
        assert set(workflow["jobs"]) == {"pytest", "pipeline-smoke", JOB}
        # PyYAML reads the `on:` key as True.
        on = workflow[True]
        assert on["push"]["branches"] == ["main", "feature/**", "fix/**", "refactor/**", "chore/**", "docs/**"]
        assert "pull_request" in on and "workflow_dispatch" in on
        for job in ("pytest", "pipeline-smoke"):
            setup = next(s for s in workflow["jobs"][job]["steps"] if "setup-python" in s.get("uses", ""))
            assert setup["with"]["python-version"] == "3.14"

    def test_job_runs_python_312_read_only(self, workflow):
        job = workflow["jobs"][JOB]
        assert job["permissions"] == {"contents": "read"}
        steps = job["steps"]
        checkout = next(s for s in steps if s.get("uses", "").startswith("actions/checkout"))
        assert checkout["with"]["persist-credentials"] is False
        setup = next(s for s in steps if "setup-python" in s.get("uses", ""))
        assert setup["with"]["python-version"] == "3.12"
        install = next(s for s in steps if s.get("name") == "Install dependencies")
        assert "pip install -r requirements-dev.txt" in install["run"]

    def test_failures_are_not_hidden(self, workflow):
        steps = workflow["jobs"][JOB]["steps"]
        assert not any("continue-on-error" in s for s in steps)
        run = next(s for s in steps if s.get("name") == "Run Player Props suites on Python 3.12")["run"]
        assert "tests/player_props_py312_suites.txt" in run
        assert "-p tests.no_network_plugin" in run and "--junitxml=" in run
        assert 'exit "$status"' in run and "status=$?" in run
        assert "| tee" not in run.split("python -m pytest", 1)[1].split("status=$?")[0]   # no pipe masks the code
        guard = run.index('if [ "${#SUITES[@]}" -eq 0 ]; then')     # an empty list must not run the default tree
        assert "exit 1" in run[guard:run.index("set +e")] and guard < run.index("python -m pytest")
        upload = next(s for s in steps if s.get("uses", "").startswith("actions/upload-artifact"))
        assert upload["if"] == "always()" and upload["with"]["path"] == "${{ runner.temp }}/py312"

    def test_versions_are_recorded(self, workflow):
        steps = workflow["jobs"][JOB]["steps"]
        names = [s.get("name") for s in steps]
        assert names.index("Record Python and library versions") < names.index("Run Player Props suites on Python 3.12")
        rec = steps[names.index("Record Python and library versions")]["run"]
        assert rec.lstrip().startswith("set -o pipefail")             # tee must not mask a failure
        for lib in ("pandas", "numpy", "scikit-learn", "xgboost", "streamlit", "pytest"):
            assert f'"{lib}"' in rec


class TestNetworkPlugin:
    """The socket guard, run in a pytest subprocess. Refused calls never reach
    the real functions (spies replace them), so these tests need no network."""

    # Conftest-free prelude: replace the saved originals with spies that record
    # calls, so "refused before resolution" is checked, not assumed.
    SPIES = (
        "import socket, pytest\n"
        "from tests import no_network_plugin as guard\n"
        "from tests.no_network_plugin import NetworkBlocked\n"
        "@pytest.fixture\n"
        "def calls(monkeypatch):\n"
        "    seen = []\n"
        "    for name in ('getaddrinfo', 'gethostbyname', 'gethostbyname_ex', 'gethostbyaddr',\n"
        "                 'create_connection'):\n"
        "        monkeypatch.setitem(guard._saved, name, lambda *a, _n=name, **k: seen.append((_n, a)))\n"
        "    return seen\n"
    )

    def run_with_plugin(self, tmp_path, body, expect_passed, may_skip=0):
        test = tmp_path / "test_net.py"
        test.write_text(self.SPIES + body, encoding="utf-8")
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                            "-p", "tests.no_network_plugin", str(test)],
                           cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
        outcomes = [f"{expect_passed} passed"] + [f"{expect_passed - n} passed, {n} skipped"
                                                  for n in range(1, may_skip + 1)]
        assert any(o in r.stdout for o in outcomes), r.stdout[-3000:]
        return r

    def test_remote_hosts_are_refused(self, tmp_path):
        self.run_with_plugin(tmp_path, (
            "def test_lookup(calls):\n"
            "    with pytest.raises(NetworkBlocked):\n"
            "        socket.getaddrinfo('github.com', 443)\n"
            "    assert calls == []\n"
            "def test_connect_ip():\n"
            "    s = socket.socket()\n"
            "    try:\n"
            "        with pytest.raises(NetworkBlocked):\n"
            "            s.connect(('93.184.216.34', 80))\n"
            "        with pytest.raises(NetworkBlocked):\n"
            "            s.connect_ex(('93.184.216.34', 80))\n"
            "    finally:\n"
            "        s.close()\n"
            "def test_create_connection(calls):\n"
            "    with pytest.raises(NetworkBlocked):\n"
            "        socket.create_connection(('93.184.216.34', 80), timeout=1)\n"
            "    assert calls == []\n"
            "@pytest.mark.parametrize('ip', ['10.0.0.1', '2001:db8::1', '::ffff:8.8.8.8', '[2001:db8::1]'])\n"
            "def test_non_loopback_literals(calls, ip):\n"
            "    with pytest.raises(NetworkBlocked):\n"
            "        socket.getaddrinfo(ip, 80)\n"
            "    assert calls == []\n"), expect_passed=7)

    def test_prefix_lookalikes_are_refused_before_resolution(self, tmp_path):
        self.run_with_plugin(tmp_path, (
            "HOSTS = ['127.example.com', '127.0.0.1.nip.io', '127.0.0.1.example.com', '127.1',\n"
            "         'localhost.example.com', 'localhostx', '::1.example.com', '0.0.0.0.example.com',\n"
            "         b'127.example.com']\n"
            "@pytest.mark.parametrize('host', HOSTS)\n"
            "def test_lookups(calls, host):\n"
            "    for fn in (socket.getaddrinfo, socket.gethostbyname, socket.gethostbyname_ex):\n"
            "        with pytest.raises(NetworkBlocked):\n"
            "            fn(host, 80) if fn is socket.getaddrinfo else fn(host)\n"
            "    assert calls == []\n"
            "def test_connect_by_name(calls):\n"
            "    s = socket.socket()\n"
            "    try:\n"
            "        with pytest.raises(NetworkBlocked):\n"
            "            s.connect(('127.example.com', 80))\n"
            "    finally:\n"
            "        s.close()\n"
            "    with pytest.raises(NetworkBlocked):\n"
            "        socket.create_connection(('127.example.com', 80), timeout=1)\n"
            "    assert calls == []\n"), expect_passed=10)

    def test_additional_dns_apis_are_guarded(self, tmp_path):
        self.run_with_plugin(tmp_path, (
            "def test_refused(calls):\n"
            "    for fn, arg in ((socket.gethostbyname, 'example.com'), (socket.gethostbyname_ex, 'example.com'),\n"
            "                    (socket.gethostbyaddr, '93.184.216.34')):\n"
            "        with pytest.raises(NetworkBlocked):\n"
            "            fn(arg)\n"
            "    assert calls == []\n"
            "def test_local_passes_through(calls):\n"
            "    socket.gethostbyname('localhost'); socket.gethostbyname_ex('127.0.0.1')\n"
            "    socket.gethostbyaddr('::1')\n"
            "    assert [n for n, _ in calls] == ['gethostbyname', 'gethostbyname_ex', 'gethostbyaddr']\n"
            ), expect_passed=2)

    def test_literal_loopback_is_allowed(self, tmp_path):
        self.run_with_plugin(tmp_path, (
            "@pytest.mark.parametrize('host', ['localhost', 'LOCALHOST', '127.0.0.1', '127.5.6.7', '::1',\n"
            "                                  '[::1]', '::ffff:127.0.0.1', '0.0.0.0', '::', None])\n"
            "def test_lookup_passes_through(calls, host):\n"
            "    socket.getaddrinfo(host, 80)\n"
            "    assert calls == [('getaddrinfo', (host, 80))]\n"
            "def test_ipv4_loopback_connection():\n"
            "    srv = socket.socket(); srv.bind(('127.0.0.1', 0)); srv.listen(8)\n"
            "    try:\n"
            "        socket.create_connection(srv.getsockname(), timeout=2).close()\n"
            "        c = socket.socket(); assert c.connect_ex(srv.getsockname()) == 0; c.close()\n"
            "    finally:\n"
            "        srv.close()\n"
            "def test_ipv6_loopback_connection():\n"
            "    try:\n"
            "        srv = socket.socket(socket.AF_INET6); srv.bind(('::1', 0))\n"
            "    except OSError:\n"
            "        pytest.skip('no IPv6 loopback on this host')\n"
            "    srv.listen(1)\n"
            "    try:\n"
            "        c = socket.socket(socket.AF_INET6); c.connect(srv.getsockname()[:2]); c.close()\n"
            "    finally:\n"
            "        srv.close()\n"
            "def test_real_localhost_lookup():\n"
            "    assert socket.getaddrinfo('localhost', 0)\n"), expect_passed=13, may_skip=1)

    def test_patched_functions_are_restored(self, tmp_path):
        """After pytest finishes, every patched function is the original again."""
        (tmp_path / "test_noop.py").write_text("def test_noop():\n    pass\n", encoding="utf-8")
        script = tmp_path / "check_restore.py"
        script.write_text(
            "import socket, sys, pytest\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "from tests import no_network_plugin as guard\n"
            "funcs = {n: getattr(socket, n) for n in guard.PATCHED}\n"
            "methods = {n: getattr(socket.socket, n) for n in guard.PATCHED_METHODS}\n"
            "seen = {}\n"
            "class Probe:\n"
            "    def pytest_sessionstart(self, session):\n"
            "        seen.update({n: getattr(socket, n) is f for n, f in funcs.items()})\n"
            "        seen.update({n: getattr(socket.socket, n) is f for n, f in methods.items()})\n"
            f"code = pytest.main(['-q', '-p', 'no:cacheprovider', '-p', 'tests.no_network_plugin',\n"
            f"                    {str(tmp_path / 'test_noop.py')!r}], plugins=[Probe()])\n"
            "assert code == 0, code\n"
            "assert len(seen) == 7 and not any(seen.values()), seen   # all patched during the run\n"
            "assert all(getattr(socket, n) is f for n, f in funcs.items())\n"
            "assert all(getattr(socket.socket, n) is f for n, f in methods.items())\n"
            "assert guard._saved == {}\n"
            "print('RESTORED')\n", encoding="utf-8")
        r = subprocess.run([sys.executable, str(script)], cwd=ROOT, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        assert r.returncode == 0 and "RESTORED" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
