"""Process-wide record of the libraries a run loaded (docs/libraries.md, "Pinning in run configurations").

A loader calls :func:`record` with the library's identity (name, file, content hash, status "pinned" with its pin set,
or "unpinned"). An output's provenance takes :func:`loaded` (or :func:`summary`), so an output never claims a pinned
library it did not run on. Standard library only.
"""
import threading

STATUSES = ("pinned", "unpinned")
_LOCK = threading.Lock()
_LOADED = []


def record(entry):
    """Add one load (a dict with at least name, content_hash and status); returns the entry."""
    if entry.get("status") not in STATUSES or not entry.get("name") or not entry.get("content_hash"):
        raise ValueError("a library load record needs name, content_hash and status 'pinned' or 'unpinned'")
    with _LOCK:
        _LOADED.append(dict(entry))
    return entry


def loaded():
    """Every load recorded in this process, in order (copies)."""
    with _LOCK:
        return [dict(e) for e in _LOADED]


def summary():
    """One entry per (name, content hash): the loads deduplicated, with "pinned" only if every load of it was."""
    out = {}
    for e in loaded():
        k = (e["name"], e["content_hash"])
        if k not in out:
            out[k] = dict(e)
        elif e["status"] != "pinned":
            out[k].update(status="unpinned", pin_set=None)
    return list(out.values())


def all_pinned():
    """True when at least one library was loaded and every load was pinned."""
    s = summary()
    return bool(s) and all(e["status"] == "pinned" for e in s)


def clear():
    """Forget the recorded loads (tests; a process running several independent jobs)."""
    with _LOCK:
        _LOADED.clear()
