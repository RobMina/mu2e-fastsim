"""crvfastsim.objective -- the CRV response objective shared by the fast chain, stock MC, data and Mu2eBO.

    crvline      CRV-only straight lines through EX, T1, T2 and the muon taggers (fit_arrays on EventNtuple-named arrays)
    crvresponse  response observables on those lines, the accumulator state, and the size-free MC-vs-data objective
    bars         barId -> (sector slot, alignment body, layer)

Only numpy, awkward (and uproot for EventNtuple files): importable by Mu2eBO kits and grid jobs with PYTHONPATH=crv.
crvextrap.crvline and crvextrap.analyses.crvresponse re-export these modules.

    python -m crvfastsim.objective NTUPLE.root [...] --emit X.json
    python -m crvfastsim.objective "dir/*.json" --merge --compare "ref/*.json" --outdir DIR [--n0 10000]
"""
