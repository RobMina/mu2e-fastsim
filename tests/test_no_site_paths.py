"""No user name and no site, user or storage path may be written into this repository.

Site paths come from configuration (site.env.example, mu2efastsim.site). Public locations stay allowed: /cvmfs,
system paths (/usr, /bin, /dev, /proc, /etc), the generic node-local cache roots /tmp, /local and /localscratch
(bare, or as a mktemp template), relative and home-relative paths, and placeholders such as /path/to.
The core/, crv/ and tracker/ submodules are separate repositories with their own check.
"""
import hashlib
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMODULES = ("core", "crv", "tracker")

# User names, matched as whole alphanumeric tokens (case-insensitive) by SHA-256, so the names never appear here.
USER_NAME_SHA256 = {"4baf0263de48597ccb579ba119bb6d2f8fea4bba418326f375a4853462ab82fa",
                    "05436ea5391c8a9c46f09d1200e8669806543032d865c86649dfacc9293603f4"}
_TOKEN = re.compile(r"[A-Za-z0-9]+")


class _UserNames:
    """search() like a compiled pattern: the first token whose hash is in USER_NAME_SHA256, or None."""
    hashes = USER_NAME_SHA256

    def search(self, line):
        for m in _TOKEN.finditer(line):
            if hashlib.sha256(m.group(0).lower().encode()).hexdigest() in self.hashes:
                return m
        return None


USER_NAMES = _UserNames()

# Absolute site, user or storage paths: the prefix must start a path (not follow a word, a dot or a variable).
_PREFIXES = (
    "exp", "pnfs", "scratch", "sfs", "lus", "eagle", "grand", "flare", "home", "nashome", "Users", "project",
    "projects", "gpfs", "lustre", "wclustre", "global", "data1", "work",
)
SITE_PATH = re.compile(r"(?<![\w.~}$\-/])/(?:" + "|".join(_PREFIXES) + r")(?=/|\b)")
# Named directories under /tmp; a bare /tmp, /tmp/* in a case pattern or a mktemp template (NAME.XXXXXX) is fine.
TMP_DIR = re.compile(r"(?<![\w.~}$\-/])/tmp/(?![\w.-]*\.XXXXXX)[A-Za-z0-9_]")
# Home-relative paths to ordinary directories (a personal layout); hidden tool state such as ~/.mu2efastsim is fine.
HOME_DIR = re.compile(r"~/(?!\.)[A-Za-z0-9_]")

SELF = Path(__file__).resolve()


def _tracked_files():
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True).stdout
        names = [n for n in out.decode().split("\0") if n and n.split("/")[0] not in SUBMODULES]
        if names:
            return [ROOT / n for n in names]
    except (OSError, subprocess.CalledProcessError):
        pass
    found = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        top = Path(dirpath) == ROOT
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", ".pytest_cache", ".venv")
                       and not (top and d in SUBMODULES)]
        found.extend(Path(dirpath) / f for f in filenames)
    return found


def _text(path):
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\0" in data[:8192]:
        return None
    return data.decode("utf-8", errors="replace")


def _violations():
    bad = []
    for path in _tracked_files():
        if path.resolve() == SELF or not path.is_file():
            continue
        text = _text(path)
        if text is None:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for rx in (USER_NAMES, SITE_PATH, TMP_DIR, HOME_DIR):
                m = rx.search(line)
                if m:
                    bad.append(f"{path.relative_to(ROOT)}:{lineno}: {m.group(0)!r} in {line.strip()[:120]}")
                    break
    return bad


def _hit(s):
    return bool(USER_NAMES.search(s) or SITE_PATH.search(s) or TMP_DIR.search(s) or HOME_DIR.search(s))


def test_patterns_catch_what_they_should():
    assert HOME_DIR.search("cd ~/mu2e/x") and not HOME_DIR.search("~/.mu2efastsim/kit")
    for s in ("/exp/mu2e/data/x", "cd /pnfs/mu2e", "'/scratch/abc'", "/lus/eagle/projects/x", "x=/home/u",
              "/Users/someone", "/project/area", "/sfs/qumulo", "/nashome/u/x",
              "out=/tmp/my_run", "Path('/tmp/thread-h')"):
        assert _hit(s), s
    for s in ("/cvmfs/mu2e.opensciencegrid.org", "/usr/bin/env", "/path/to/scratch", "terminal", "determinant",
              "$HOME/exp/x", "${SCRATCH}/scratch/y", "relative/exp/z", "https://example.org/projects/x", "/tmp",
              "mktemp -d /tmp/offline-probe.XXXXXXXX", "/tmp|/tmp/*|/local|/localscratch", "/dev/shm/x",
              "~/.mu2efastsim/kit"):
        assert not _hit(s), s
    names = _UserNames()
    names.hashes = {hashlib.sha256(b"alice").hexdigest()}
    assert names.search("user ALICE here") and names.search("/data/alice/x") and not names.search("malice")


def test_submodules_excluded():
    assert not any(p.relative_to(ROOT).parts[0] in SUBMODULES for p in _tracked_files())


def test_no_user_names_or_site_paths_in_tracked_files():
    bad = _violations()
    assert not bad, "site paths or user names in the repository:\n" + "\n".join(bad)
