"""The magnetic field that the stock trigger finders read, for the device ports that need it.

Stock helix finders take the field z at the detector origin (the tracker centre, (-solenoidOffset, 0,
detectorSystemZ0) = (-3904, 0, 10171) mm) from the BFieldManager once per run. For the replay geometry of the
trigger validation (Offline v13_41_01, geom_run1_a with the field map bfgeom_no_tsu_ps_v01) that is
Bz = 1.0117056584464001 T: the stock BFieldManager built as GeometryService builds it, printed with full double
precision. The stock BFieldTest module gives the same value to its 10 printed decimals (1.0117056584; 1 mm away along
z: 1.0117126903 / 1.0116986189 T). The 10-decimal value is not enough: CalHelixFinder's seed t0 (a double) then
differs from stock in the last bit for about one seed in ten.

Each port keeps its stock module's precision when it uses the value: e.g. CalHelixFinderAlg::bz() is a float (the
radius window), while CalHelixFinder's `_bz0` is a double (momentum, beta and the seed t0).
"""
BZ0 = 1.0117056584464001                     # T, BFieldManager z at the detector origin, full double precision
