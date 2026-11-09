# albedo_run1a_v1.npz

Downstream-albedo library of `trkfastsim.sim.albedo` (the default of `beamfastsim.cetracker.DeviceChain`), built with
`albedo.build` from the dumper tables of 13 Geant4 CeEndpoint files (`dts.mu2e.CeEndpoint.MDC2025ba`, sequencers
001430_00000000..0600 and 1400..1900; Run 1A geometry), at the virtual detector at the tracker's downstream end
(VD 15, z = 11,810.5 mm):

- 36,649 exits of the conversion electron, binned by exit radius, pz/p and the sign of the radial momentum;
- 1,305 exits with returns, 1,550 upstream electron crossings (370 of them the conversion electron itself): returns
  from 0.96 % of the exits for the electron itself, 3.6 % for any electron.

Held-out sequencers 001430_00000700..1300 were not used. Per return: azimuth and radius relative to the exit,
momentum (radial, azimuthal, axial), delay, and whether it is the conversion electron.
