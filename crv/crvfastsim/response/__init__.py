"""Fast CRV response: photons (port of Offline CRVResponse MakeCrvPhotons) and, next, SiPM charges.

The CRV lookup tables (CRVConditions v6_0) are Mu2e-internal: they are read at run time from $CRVFS_LOOKUP_DIR
(default: the Mu2e CVMFS DataFiles area) and are never committed or packed into a public image.
"""
