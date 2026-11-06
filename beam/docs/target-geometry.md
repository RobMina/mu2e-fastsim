# Target geometry primitives

`target_geometry_jax` contains Mu2e target-layer setup shared by device transport
callers: material/thickness types, field-map loading and first-containing-map
selection, kinematics, fixed-shape gathers, foil arrays, radial coordinates,
local frames and bracketed shell bisection. `cechain_jax` calls these functions;
its execution loop, refill, random history and transport defaults stay separate.

Actual foil identity is distinct from a material/thickness type. Foil frames
currently use axial-face origins. The shell root finder requires a selected
bracket: it is not a general earliest-crossing query. These helpers do not enable
finite charged-foil transport or a shower model.

`flash_exit` supplies independent straight photon-ray intersections with axial
faces, rims and holes. `flash_annulus_jax` re-exports the core restricted charged
annulus primitive without duplicating its implementation. The latter requires
float64 and a straight or axial-field trajectory; unresolved results must be
handled before transport, not replaced by a clear-path infinity.

The slab birth-record collector, stock input-state reader and Al attenuation
table are independent opt-in primitives. See [flash-source.md](flash-source.md)
and [flash-attenuation.md](flash-attenuation.md). Together these are building
blocks, not a prediction of the tracker flash observable.
