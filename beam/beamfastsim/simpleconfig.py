"""Minimal reader of Mu2e SimpleConfig geometry files (Offline/Mu2eG4/geom/*.txt), following #include.

Later definitions override earlier ones, as in SimpleConfig. Values: numbers, strings, bools and {...} vectors.
"""
import os
import re

_STMT = re.compile(r'(?:vector<\s*\w+\s*>|double|int|bool|string|float)\s+([\w.]+)\s*=\s*(.*?);', re.S)
_INC = re.compile(r'^\s*#include\s+"([^"]+)"', re.M)


def _strip_comments(text):
    return re.sub(r'//[^\n]*', '', text)


def _expand(path, root, seen, source_files=None):
    path = os.path.realpath(path)
    if path in seen:
        return ""
    seen.add(path)
    if source_files is not None:
        source_files.append(path)
    with open(path) as f:
        text = _strip_comments(f.read())
    out, pos = [], 0
    for m in _INC.finditer(text):
        out.append(text[pos:m.start()])
        inc = os.path.join(root, m.group(1))
        if not os.path.exists(inc):
            inc = os.path.join(os.path.dirname(path), os.path.basename(m.group(1)))
        out.append(_expand(inc, root, seen, source_files))
        pos = m.end()
    out.append(text[pos:])
    return "\n".join(out)


def _value(s):
    s = s.strip()
    if s.startswith("{"):
        return [_value(x) for x in s.strip("{} \n").split(",") if x.strip()]
    if s.startswith('"'):
        return s.strip('"')
    if s in ("true", "false"):
        return s == "true"
    try:
        return float(s)
    except ValueError:
        return s


def read(path, root, source_files=None):
    """Dict of all parameters of geometry file `path`; #include paths ("Offline/...") resolve under `root`.

    An #include that is not under `root` is looked up next to the including file (geometry overlays)."""
    text = _expand(path, root, set(), source_files)
    return {m.group(1): _value(m.group(2)) for m in _STMT.finditer(text)}
