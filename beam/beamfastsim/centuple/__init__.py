"""Fast CE chain -> stock EventNtuple files (opt-in).

Three stages, each runnable on its own:

1. ``record`` (GPU, JAX): births -> ``cetracker.DeviceChain`` (tracker transport, device digis, reco response) ->
   the conversion electron's truth at TT_Front / TT_Mid / TT_Back -> calorimeter first contact and clusters ->
   one fast record (``.npz``) per batch, plus the device digis for the trigger stage.
2. ``trigger`` (GPU, JAX, its own process: the trigger flips JAX's process-wide x64 flag in float32): device digis
   -> the tracker trigger paths (``trigger.device``) -> decisions added to the record.
3. ``write`` (CPU, PyROOT in the Mu2e environment that holds the EventNtuple dictionaries): record + donor library
   -> ``EventNtuple/ntuple`` as a split TTree with the stock branch layout, ``EventNtuple/n_proc_events`` and a
   provenance object.

The branch contract (what the analysis server's EventNtuple analyses and pyutils read) and the coverage of each
branch by the fast chain are in ``contract.py`` and ``beam/docs/eventntuple.md``.
"""
