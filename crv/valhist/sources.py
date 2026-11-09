"""Historical grid sample locations, from configuration (see site.env.example).

  CRVFS_PNFS    dCache area that holds this project's grid samples (g4ref_v1/, crygen_v1/, ...)
  CRVFS_OWNER   owner field of those Mu2e file names (dts.<owner>.<description>...)

A missing variable raises mu2efastsim.site.SiteConfigError naming it.
"""
from mu2efastsim import site


def pnfs():
    return site.require("CRVFS_PNFS").rstrip("/")


def owner():
    return site.require("CRVFS_OWNER")


def g4ref_art(tier):
    """Template (field {sequence}) of a g4ref_v1 art file of data tier 'dts' or 'gen'."""
    return f"{pnfs()}/g4ref_v1/{tier}.{owner()}.CosmicCRYExtractedRef.v0.{{sequence}}.art"


def crygen_art():
    """Template (field {subrun}) of a crygen_v1 generator art file."""
    return f"{pnfs()}/crygen_v1/gen.{owner()}.CosmicCRYExtractedGen.v1.001400_{{subrun:08d}}.art"


def crygen_log():
    """Template (field {subrun}) of a crygen_v1 grid job log."""
    return f"{pnfs()}/crygen_v1/logs/mu2e.001400_{{subrun:08d}}.log.gz"
