"""Commit the staged changes and publish them to a branch on origin, safely.

Used by the nightly workflow's two publication steps ("Commit and push
changes" and "Commit best_bets_today.json"). The step stages its files, then
runs this script, which:

1. Treats an empty index as a successful no-op ("nothing to commit"); nothing
   is committed or pushed.
2. Commits the staged files once.
3. Pushes that commit to ``origin/<branch>`` without force, then confirms it is
   on the branch by fetching it back. Publication is reported only once the
   commit is confirmed there.
4. After a failed push, checks the branch first. If the commit is already
   there (the server accepted it but the client saw an error), that counts as
   published, so the commit is never pushed twice. Transient failures (HTTP
   5xx, timeouts, dropped connections) are retried a bounded number of times
   with a short, growing delay.
5. If the branch moved on, rebuilds the commit on top of it, but only when
   the remote changes touch none of the paths the commit changes and none of
   the ``--guard`` paths (files the commit's contents were validated or built
   against). The rebuilt commit carries byte-for-byte the same files as the
   original and leaves every remote change as it was; both facts are checked
   before it is pushed. Any overlap stops with a visible failure. Nothing is
   merged by content, regenerated or overwritten.
6. Stops at once on errors a retry cannot fix (authentication, permissions,
   protected branches, a missing repository).

Every outcome is one of three:
- **published**: a fetch showed the commit on the branch (exit 0);
- **not published**: nothing was pushed, the server refused the push, or a
  fetch after the last push showed the commit absent (exits 3-7);
- **unconfirmed** (exit 8): a push may have landed, but no later fetch could
  check it ("Publication unconfirmed; the commit may already be on
  origin/main."). The message gives the full commit SHA, remote and branch so
  the outcome can be checked read-only (``git ls-remote``). Nothing is rolled
  back, deleted or force-pushed.

Exit codes: 0 published or nothing to commit, 2 usage, 3 staging or commit
error, 4 permanent push error, 5 conflicting remote changes, 6 a rebuilt
commit failed its checks, 7 retries exhausted, 8 unconfirmed. Every failure
keeps the local commit. The remote branch is never rewritten, but after an
unconfirmed outcome it may already hold the commit.

In GitHub Actions it also writes ``result`` (``published``,
``nothing-to-commit``, ``not-published`` or ``unconfirmed``) and ``commit`` to
``$GITHUB_OUTPUT``, a section to the job summary, and an ``::error``
annotation on failure.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

EXIT_USAGE, EXIT_COMMIT, EXIT_PERMANENT, EXIT_CONFLICT, EXIT_INTEGRATION, EXIT_EXHAUSTED, EXIT_UNCONFIRMED =     2, 3, 4, 5, 6, 7, 8
PUBLISHED, NOTHING, NOT_PUBLISHED, UNCONFIRMED = "published", "nothing-to-commit", "not-published", "unconfirmed"
UNCONFIRMED_MSG = "Publication unconfirmed; the commit may already be on {remote}/{branch}."

GIT_TIMEOUT = 180  # seconds for any one git command; a hung push is treated as transient
REMOTE_REF = "refs/publish-commit/remote"  # private ref holding the fetched branch tip

# Errors no retry can fix. Anything else (HTTP 5xx, timeouts, "remote end hung
# up", "[remote rejected] ... (Internal Server Error)") is treated as transient.
PERMANENT = re.compile(
    r"authentication failed|could not read (username|password)|terminal prompts disabled"
    r"|invalid username or password|permission to \S+ denied|write access to repository not granted"
    r"|returned error: 40[134]\b|repository not found|does not appear to be a git repository"
    r"|protected branch|hook declined|\bGH0\d\d\b",
    re.IGNORECASE,
)
# The client refused before sending anything: the push did not land.
REJECTED = re.compile(r"\[rejected\]|non-fast-forward|\(fetch first\)")
SECRET = re.compile(r"(://)[^/@\s]+@|\b(gh[pousr]_|github_pat_)\w+")


class PublishError(Exception):
    def __init__(self, code: int, message: str, commit: str = ""):
        super().__init__(message)
        self.code, self.commit = code, commit

    @property
    def status(self) -> str:
        return UNCONFIRMED if self.code == EXIT_UNCONFIRMED else NOT_PUBLISHED


def _unconfirmed(commit: str, remote: str, branch: str, why: str) -> PublishError:
    return PublishError(EXIT_UNCONFIRMED, "\n".join([
        UNCONFIRMED_MSG.format(remote=remote, branch=branch),
        f"  commit: {commit}", f"  remote: {remote}", f"  branch: {branch}", f"  why: {why}",
        f"  check (read-only): git ls-remote {remote} refs/heads/{branch}, then, after"
        f" git fetch {remote} {branch}: git merge-base --is-ancestor {commit} FETCH_HEAD"]), commit)


def redact(text: str) -> str:
    return SECRET.sub(lambda m: (m.group(1) or "") + "***" + ("@" if m.group(1) else ""), text)


class Git:
    """Runs git in one repository. Tests override ``push`` to inject failures."""

    def __init__(self, cwd: Path | str = ".", timeout: float = GIT_TIMEOUT):
        self.cwd, self.timeout = Path(cwd), timeout
        self.env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}

    def run(self, *args: str, input: str | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
        # Binary I/O: text mode on Windows would turn "\n" in stdin into "\r\n".
        try:
            r = subprocess.run(["git", *args], cwd=self.cwd, capture_output=True, timeout=self.timeout,
                               input=None if input is None else input.encode("utf-8"),
                               env={**self.env, **(env or {})})
        except subprocess.TimeoutExpired as e:
            return subprocess.CompletedProcess(e.cmd, 124, (e.stdout or b"").decode("utf-8", "replace"),
                                               f"git {args[0]} timed out after {self.timeout:g}s")
        return subprocess.CompletedProcess(r.args, r.returncode, r.stdout.decode("utf-8", "replace"),
                                           r.stderr.decode("utf-8", "replace"))

    def out(self, *args: str, **kw) -> str:
        r = self.run(*args, **kw)
        if r.returncode:
            raise PublishError(EXIT_INTEGRATION, f"git {' '.join(args)} failed: {redact(r.stderr.strip())}")
        return r.stdout

    def push(self, remote: str, refspec: str) -> subprocess.CompletedProcess:
        return self.run("push", "--porcelain", remote, refspec)

    def fetch(self, remote: str, branch: str) -> subprocess.CompletedProcess:
        return self.run("fetch", "--quiet", "--no-tags", "--update-shallow", remote,
                        f"+refs/heads/{branch}:{REMOTE_REF}")

    def is_ancestor(self, a: str, b: str) -> bool:
        return self.run("merge-base", "--is-ancestor", a, b).returncode == 0

    def changed(self, a: str, b: str) -> set[str]:
        """Paths whose content differs between two commits' trees."""
        return {p for p in self.out("diff", "--name-only", "--no-renames", "-z", a, b).split("\0") if p}


def _indent(text: str) -> str:
    return "\n".join("    " + line for line in text.strip().splitlines()) or "    (no output)"


def _rebuild(git: Git, target: str, remote_sha: str, guards: set[str], log) -> str:
    """The commit ``target`` replayed onto ``remote_sha``, or PublishError.

    Allowed only when the remote changes since ``target``'s parent are disjoint
    from what ``target`` changes and from the guarded paths."""
    parents = git.out("rev-list", "--parents", "-n", "1", target).split()[1:]
    if len(parents) != 1:
        raise PublishError(EXIT_INTEGRATION, f"cannot rebuild {target[:12]}: it has {len(parents)} parents")
    base = parents[0]
    ours, theirs = git.changed(base, target), git.changed(base, remote_sha)
    log(f"origin moved on: {len(theirs)} path(s) changed remotely since {base[:12]}; this commit changes {len(ours)}")
    blocked = sorted((ours | guards) & theirs)
    if blocked:
        raise PublishError(EXIT_CONFLICT,
                           "origin changed path(s) this publication changes or depends on, so it was not "
                           "published and neither side was changed: " + ", ".join(blocked))
    entries = {}
    for rec in git.out("ls-tree", "-r", "-z", "--full-tree", target).split("\0"):
        if rec:
            meta, path = rec.split("\t", 1)
            entries[path] = meta
    zero = "0" * len(target)
    lines = []
    for path in sorted(ours):
        if path in entries:
            mode, _, sha = entries[path].split()
            lines.append(f"{mode} {sha}\t{path}")
        else:
            lines.append(f"0 {zero}\t{path}")
    with tempfile.TemporaryDirectory() as tmp:
        idx = {"GIT_INDEX_FILE": str(Path(tmp) / "index")}
        git.out("read-tree", remote_sha, env=idx)
        git.out("update-index", "--index-info", input="\n".join(lines) + "\n", env=idx)
        tree = git.out("write-tree", env=idx).strip()
    name, email, date = git.out("log", "-1", "--format=%an%x00%ae%x00%ad", "--date=raw", target).rstrip("\n").split("\0")
    message = git.out("log", "-1", "--format=%B", target)
    new = git.out("commit-tree", tree, "-p", remote_sha, input=message,
                  env={"GIT_AUTHOR_NAME": name, "GIT_AUTHOR_EMAIL": email, "GIT_AUTHOR_DATE": date}).strip()
    # The rebuilt commit must change exactly our paths relative to the remote,
    # and differ from the original exactly in the remote's paths.
    if git.changed(remote_sha, new) != ours or git.changed(target, new) != theirs:
        raise PublishError(EXIT_INTEGRATION, f"rebuilt commit {new[:12]} failed its content checks; not published")
    log(f"rebuilt as {new[:12]} on top of {remote_sha[:12]}: same {len(ours)} file(s), remote changes kept")
    return new


def publish(git: Git, *, message: str, remote: str = "origin", branch: str = "main", guards=(),
            attempts: int = 4, delay: float = 10.0, sleep=time.sleep, log=print) -> tuple[str, str]:
    """Commit the index and publish it. Returns (result, commit sha)."""
    staged = git.run("diff", "--cached", "--quiet")
    if staged.returncode == 0:
        log("Nothing to commit: no staged changes. Nothing was published.")
        return NOTHING, ""
    if staged.returncode != 1:
        raise PublishError(EXIT_COMMIT, f"could not inspect the staged changes: {redact(staged.stderr.strip())}")
    c = git.run("commit", "--quiet", "-m", message)
    if c.returncode:
        raise PublishError(EXIT_COMMIT, f"git commit failed (exit {c.returncode}):\n{_indent(redact(c.stdout + c.stderr))}")
    target = git.out("rev-parse", "HEAD").strip()
    log(f"Committed {target[:12]}: {message}")
    guards = set(guards)

    # True while a push since the last successful read-back may have landed:
    # only a fetch can then say whether the commit is on the branch.
    unchecked = False
    for attempt in range(1, attempts + 1):
        log(f"Attempt {attempt}/{attempts}: pushing {target[:12]} to {remote}/{branch}")
        r = git.push(remote, f"{target}:refs/heads/{branch}")
        output = redact(r.stdout + r.stderr)
        if r.returncode:
            log(f"  push failed (exit {r.returncode}):\n{_indent(output)}")
            if PERMANENT.search(output):
                if unchecked:
                    raise _unconfirmed(target, remote, branch, "an earlier push could not be checked, and this "
                                       f"one failed with an error a retry cannot fix:\n{_indent(output)}")
                raise PublishError(EXIT_PERMANENT, f"the server refused the push with an error a retry cannot fix; "
                                   f"{target} was not published:\n{_indent(output)}", target)
            unchecked = unchecked or not REJECTED.search(output)
        else:
            log("  push reported success; confirming on the remote")
            unchecked = True
        f = git.fetch(remote, branch)
        if f.returncode:
            fout = redact(f.stdout + f.stderr)
            log(f"  could not read {remote}/{branch} back (exit {f.returncode}):\n{_indent(fout)}")
            if PERMANENT.search(fout):
                if unchecked:
                    raise _unconfirmed(target, remote, branch, "reading the branch back failed with an error "
                                       f"a retry cannot fix:\n{_indent(fout)}")
                raise PublishError(EXIT_PERMANENT, f"fetch failed with an error a retry cannot fix; {target} was "
                                   f"not published:\n{_indent(fout)}", target)
        else:
            unchecked = False
            remote_sha = git.out("rev-parse", REMOTE_REF).strip()
            if git.is_ancestor(target, remote_sha):
                note = " (the server had accepted it despite the reported error)" if r.returncode else ""
                log(f"Published: {target} is on {remote}/{branch} (tip {remote_sha[:12]}){note}.")
                return PUBLISHED, target
            if not git.is_ancestor(remote_sha, target):
                try:
                    target = _rebuild(git, target, remote_sha, guards, log)
                except PublishError as e:
                    e.commit = e.commit or target
                    raise
                continue  # push the rebuilt commit right away
            log(f"  {remote}/{branch} is still {remote_sha[:12]}; {target[:12]} is not on it")
        if attempt < attempts:
            wait = delay * attempt
            log(f"  retrying in {wait:g}s")
            sleep(wait)
    if unchecked:
        raise _unconfirmed(target, remote, branch, f"after {attempts} attempts, {remote}/{branch} could not be "
                           "read back to check the last push")
    raise PublishError(EXIT_EXHAUSTED, f"gave up after {attempts} attempts: {target} is not on {remote}/{branch} "
                       "(checked by fetching it); not published", target)


def _append(env_var: str, text: str) -> None:
    path = os.environ.get(env_var)
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--message", "-m", required=True, help="commit message")
    ap.add_argument("--label", default="Publication", help="name used in the job summary and annotations")
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--guard", action="append", default=[], metavar="PATH",
                    help="repo path the commit depends on; a remote change to it blocks publication")
    ap.add_argument("--attempts", type=int, default=int(os.environ.get("PUBLISH_ATTEMPTS", 4)))
    ap.add_argument("--delay", type=float, default=float(os.environ.get("PUBLISH_RETRY_DELAY", 10)),
                    help="seconds before retry n is n * delay")
    args = ap.parse_args(argv)
    if args.attempts < 1 or args.delay < 0:
        ap.error("--attempts must be >= 1 and --delay >= 0")
    try:
        result, sha = publish(Git(), message=args.message, remote=args.remote, branch=args.branch,
                              guards=args.guard, attempts=args.attempts, delay=args.delay,
                              log=lambda s: print(s, flush=True))
    except PublishError as e:
        unconfirmed = e.status == UNCONFIRMED
        heading = "publication UNCONFIRMED" if unconfirmed else "NOT published"
        print(f"{args.label}: {heading}: {e}", file=sys.stderr, flush=True)
        if os.environ.get("GITHUB_ACTIONS") == "true":
            title = f"{args.label} publication unconfirmed" if unconfirmed else f"{args.label} was not published"
            first = str(e).splitlines()[0]
            if e.commit and e.commit not in first:
                first += f" Commit {e.commit} ({args.remote}/{args.branch})."
            print(f"::error title={title}::{first}", flush=True)
        _append("GITHUB_OUTPUT", f"result={e.status}\ncommit={e.commit}\n")
        _append("GITHUB_STEP_SUMMARY", f"\n### {args.label}: {heading}\n\n```\n{e}\n```\n")
        return e.code
    _append("GITHUB_OUTPUT", f"result={result}\ncommit={sha}\n")
    _append("GITHUB_STEP_SUMMARY", f"\n### {args.label}: " +
            (f"published `{sha}` to {args.remote}/{args.branch}\n" if sha else "nothing to commit\n"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
