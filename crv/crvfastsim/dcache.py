"""The site's dCache namespace, from configuration (see site.env.example); nothing here names a site path.

  CRVFS_DCACHE   mount point of the dCache namespace that holds the experiment area (<mount>/mu2e/...)
  CRVFS_XROOTD   xrootd URL prefix that serves the same namespace: CRVFS_XROOTD + p[len(CRVFS_DCACHE):] reads p

A missing variable raises mu2efastsim.site.SiteConfigError naming it.
"""
import os

from mu2efastsim import site

EXPERIMENT = "mu2e"


def mount():
    """dCache mount point (CRVFS_DCACHE), without a trailing slash."""
    return site.require("CRVFS_DCACHE").rstrip("/")


def area():
    """The experiment's area in the dCache namespace (<mount>/mu2e)."""
    return mount() + "/" + EXPERIMENT


def contains(path):
    """True for a path under the dCache mount (a lexical test: nothing is looked up)."""
    return os.fspath(path).startswith(mount() + "/")


def xrootd(path):
    """xrootd URL of a dCache path (lexical: the mount prefix is replaced by CRVFS_XROOTD)."""
    path, root = os.fspath(path), mount()
    if not path.startswith(root + "/"):
        raise ValueError(f"not a dCache path: {path}")
    return site.require("CRVFS_XROOTD").rstrip("/") + path[len(root):]


def token_scope(path):
    """Storage-token scope of a path in the experiment area (<area>/x -> /mu2e/x)."""
    path, root = os.fspath(path), area()
    if not path.startswith(root + "/"):
        raise ValueError(f"not in the experiment's dCache area: {path}")
    return "/" + EXPERIMENT + path[len(root):]
