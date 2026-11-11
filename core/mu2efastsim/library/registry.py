"""Shipped names, registries and release assets of library files (docs/libraries.md, "Storage and distribution").

Small libraries ship in git next to their package (``<package>/data/``) and are resolved by name. Large or
per-configuration libraries are not in git: a registry JSON maps an identifier to the file name, its sha256 and the
URL of a GitHub release asset, and a local cache holds the downloaded files.

Registry file: ``{"schema": 1, "entries": {id: entry}}``, or a bare ``{id: entry}`` mapping (the layout of the
tracker response registry, which this reads unchanged). An entry has at least ``file`` and ``sha256`` (of the file
bytes) and usually ``url``; ``name``, ``version``, ``content_hash`` and ``config_hash`` let :meth:`Registry.find`
resolve ``name@version`` or a hash prefix as well as the id. The sha256 is checked on every resolution, whether the
file was just downloaded or already cached; a mismatch is refused, never re-downloaded silently.

Cache: ``$MU2EFS_LIBRARY_CACHE`` (or the registry's own environment variable), else
``~/.cache/mu2efastsim/libraries``. Release asset URL convention for new libraries: ``RELEASE_URL.format(tag=..., file=...)``
with tag ``lib-<name>-v<version>``.
"""
import json
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .hashing import file_sha256

SCHEMA = 1
CACHE_ENV = "MU2EFS_LIBRARY_CACHE"
RELEASE_URL = "https://github.com/RobMina/mu2e-fastsim/releases/download/{tag}/{file}"


def release_tag(name, version):
    return f"lib-{name}-v{int(version)}"


def release_url(name, version, file):
    return RELEASE_URL.format(tag=release_tag(name, version), file=file)


class RegistryError(LookupError):
    """An identifier that resolves to nothing, to several entries, or to a file whose sha256 differs."""


def default_cache(env=CACHE_ENV, subdir=""):
    base = os.environ.get(env) or os.environ.get(CACHE_ENV)
    return Path(base) if base else Path.home() / ".cache" / "mu2efastsim" / "libraries" / subdir


@dataclass
class Registry:
    """Shipped names, registry entries and a cache directory, resolved in that order."""
    entries: dict = field(default_factory=dict)
    named: dict = field(default_factory=dict)       # shipped name -> path in git
    cache: Path = None
    path: Path = None

    @classmethod
    def read(cls, path, *, named=None, cache=None):
        """A registry from its JSON file (a missing file gives an empty registry)."""
        p = Path(path)
        raw = json.loads(p.read_text()) if p.exists() else {}
        if "entries" in raw and "schema" in raw:
            if raw["schema"] != SCHEMA:
                raise RegistryError(f"{p}: registry schema {raw['schema']}, expected {SCHEMA}")
            raw = raw["entries"]
        for k, e in raw.items():
            if not isinstance(e, dict) or "file" not in e or "sha256" not in e:
                raise RegistryError(f"{p}: entry {k!r} needs file and sha256")
        return cls(entries=raw, named=dict(named or {}), cache=Path(cache) if cache else default_cache(), path=p)

    def find(self, ident):
        """Registry id of ``ident``: an id, ``name@version``, ``name`` (highest version), or a unique prefix (at
        least 8 hex digits) of an id, a content hash or a config hash."""
        s = str(ident)
        if s in self.entries:
            return s
        if "@" in s:
            name, version = s.rsplit("@", 1)
            hits = [k for k, e in self.entries.items() if e.get("name") == name and str(e.get("version")) == version]
        else:
            hits = [k for k, e in self.entries.items() if e.get("name") == s]
            if len(hits) > 1:
                best = max(int(self.entries[k].get("version", 0)) for k in hits)
                hits = [k for k in hits if int(self.entries[k].get("version", 0)) == best]
            if not hits and len(s) >= 8 and all(c in "0123456789abcdef" for c in s):
                hits = [k for k, e in self.entries.items()
                        if any(str(v).startswith(s) for v in (k, e.get("content_hash", ""), e.get("config_hash", "")))]
        if len(hits) != 1:
            raise RegistryError(f"{ident!r}: {'no' if not hits else len(hits)} registry entries match")
        return hits[0]

    def fetch(self, ident, *, download=True):
        """Local path of a registered file, downloaded into the cache when missing (``download``); sha256 checked."""
        key = self.find(ident)
        e = self.entries[key]
        p = Path(self.cache) / e["file"]
        if not p.exists():
            if not download or not e.get("url"):
                raise RegistryError(f"{key}: {p} is not in the cache" + ("" if e.get("url") else " and has no URL"))
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.name + ".part")
            with urllib.request.urlopen(e["url"]) as r, open(tmp, "wb") as f:
                for block in iter(lambda: r.read(1 << 20), b""):
                    f.write(block)
            if file_sha256(tmp) != e["sha256"]:
                tmp.unlink()
                raise RegistryError(f"{key}: the download from {e['url']} has another sha256 than the registry entry")
            tmp.replace(p)
        if file_sha256(p) != e["sha256"]:
            raise RegistryError(f"{p}: sha256 differs from the registry entry {key}")
        return p

    def resolve(self, spec, *, download=True):
        """Path of a library: a shipped name, an existing file, or a registry identifier (see :meth:`find`)."""
        if isinstance(spec, str) and spec in self.named:
            return Path(self.named[spec])
        if isinstance(spec, (str, os.PathLike)) and Path(spec).exists():
            return Path(spec)
        try:
            return self.fetch(spec, download=download)
        except RegistryError as exc:
            raise FileNotFoundError(f"library {spec!r}: not a shipped name ({sorted(self.named)}), an existing "
                                    f"file or a registry entry ({exc})") from None

    def entry(self, path, *, name=None, version=None, url=None, **extra):
        """A registry entry for a library file: its file name, sha256 and URL (release asset by default), plus the
        manifest's name, version, content hash and config hash when the file has a manifest."""
        from .store import LibraryError, load
        p = Path(path)
        e = {"file": p.name, "sha256": file_sha256(p)}
        try:
            m = load(p).manifest
            e.update(name=m["name"], version=m["version"], content_hash=m["content_hash"])
            if m.get("config_hash"):
                e["config_hash"] = m["config_hash"]
        except LibraryError:
            pass
        if name is not None:
            e["name"] = name
        if version is not None:
            e["version"] = int(version)
        if url is None and e.get("name") and e.get("version"):
            url = release_url(e["name"], e["version"], p.name)
        if url:
            e["url"] = url
        e.update(extra)
        return e

    def add(self, key, entry):
        if key in self.entries and self.entries[key] != entry:
            raise RegistryError(f"{key} is already registered with another entry")
        self.entries[key] = entry

    def write(self, path=None):
        p = Path(path or self.path)
        p.write_text(json.dumps({"schema": SCHEMA, "entries": self.entries}, indent=1, sort_keys=True) + "\n")
        return p
