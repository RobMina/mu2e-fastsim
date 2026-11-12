# Response and interaction libraries: one core, one vocabulary

Every tabulated, pooled or fitted artifact that a fast stage loads and queries in place of a simulation step is a
**library**: the CRV photon tables and secondary library, the calorimeter digi response and shower library, the
tracker response tables, row pools, albedo and delta-ray libraries, the flash response, the geometry-portable
interaction banks, and the beam tables. Each subsystem keeps its own libraries in its own package. All of them are
built on the shared core `mu2efastsim.library` and follow the conventions below, so that a library can be checked,
cited, distributed and compared in the same way whichever subsystem owns it.

The core is NumPy and the standard library only; the JAX helpers import JAX when called. Modules:

| Module | What it holds |
|---|---|
| `hashing` | canonical JSON, file sha256, array content hash, short digests |
| `axes` | `Axis` (key, unit, frame, edges, out-of-domain policy), `Grid` (row-major cells), NumPy and JAX bin lookup |
| `domain` | `OutOfDomain` (loud refusal) and `DomainCounter` (declared clips and fallbacks, counted) |
| `manifest` | the manifest schema and its checks |
| `store` | `.npz` files with an embedded or sidecar manifest; `load()` and `verify()` check everything they can |
| `registry` | shipped names, registry JSON, release assets and the download cache |
| `dtypes` | the float32/float64 compute policy and device copies |
| `fidelity` | status levels, metrics, criteria, validation records and the performance entry |

`python3 -m mu2efastsim.library check|describe|hash FILE` runs the load checks, prints the manifest, or prints the
file sha256 and the content hash.

## 1. Contract

A library module exposes the same small surface, whatever it tabulates:

```python
lib = load(spec)                       # spec: a shipped name, a file, or a registry identifier; checked (section 5)
out = lookup_numpy(lib, keys...)       # deterministic answer (expected value, interpolation), host, float64
out = sample_numpy(lib, rng, keys...)  # one draw per row, host: rng is a numpy.random.Generator
dl  = to_device(lib, dtype)            # device copy: floating arrays in dtype, indices int32
out = lookup(dl, keys...)              # device twin of lookup_numpy (jit-able)
out = sample(dl, key, keys...)         # device twin of sample_numpy: key is a jax.random key
counter = domain_counts(lib, keys...)  # DomainCounter: what was clipped, refused or served by a fallback
```

A library has `lookup*`, `sample*` or both, as its kind requires (section 3). The rules:

- **Batch shapes.** Keys are arrays with a leading row axis `[n]` (vectors `[n, 3]`); every output has the same
  leading axis, one answer per row. A row that is not a lookup (no tracker entry, a padded lane) is selected by a
  boolean mask `[n]` and gets a declared null answer (0 or NaN, named in the manifest `outputs`). Device functions
  take fixed shapes: padding and capacity are the caller's, and an overflow is reported, never truncated silently.
- **Dtypes.** Floating arrays are stored as built: float64 unless the manifest declares a smaller storage dtype for
  size (float32 pools, uint8 weights). Host code computes in float64 and returns int64 indices. Device code computes
  in the dtype the caller passes, **float32 or float64; every library supports both** and lists them in
  `dtype.compute`. Floating tables are cast once, when the device copy is made (`dtypes.device_arrays`); index arrays
  become int32 and a table that does not fit int32 is refused. `dtypes.compute_dtype(None)` is the run's device float
  (`MU2EFS_DEVICE_FLOAT`, see `mu2efastsim.precision`). Bin edges are cast to the key dtype, so a float32 key within
  one float32 step of an edge can fall in the neighbouring bin compared with float64; the float32-vs-float64
  comparison of the validation (section 6) covers that.
- **Random numbers.** Host functions take a `numpy.random.Generator` argument; device functions take a
  `jax.random` key. No library seeds itself, reads global random state or keeps a stream between calls. The number
  of draws per row is fixed and declared (`rng.draws_per_row`), so that turning an option on does not move the draws
  of others. A library that needs several streams derives them inside, with `jax.random.split` / `fold_in` on the
  device or `Generator(bit_generator.jumped())` on the host, and declares them in `rng.streams`; the caller derives
  the key it passes with `fold_in(chain_key, <stream id>)` and the chain documents its stream ids. Host and device
  twins draw the same distribution, not the same numbers.
- **Refusal before drawing.** A row that the library refuses (section 5) raises before any random number is drawn,
  so a refusal never shifts the draws of the other rows.

## 2. Description: keys, axes, units, frames

Every key of a library is an `Axis`: a name (the keyword the lookup takes), a unit (`"mm"`, `"MeV/c"`, `"ns"`,
`"1"` for dimensionless), a frame or definition (`"Mu2e"`, `"tracker entry helix"`, `"bar-local, SiPM 0"`,
`"disk-entry incident frame"`), the edges (stored in the library under `Axis.array`), how the edges were chosen
(`scale`: linear, log, irregular, quantile, categorical) and the out-of-domain policy.

- **Bins.** Bin `k` is `edges[k] <= v < edges[k + 1]` (`searchsorted(..., side="right") - 1`), one convention for
  every library; `axes.locate` and `axes.locate_jax` give the same indices for the same edges and values in the
  same dtype. The last edge belongs to the overflow side. A NaN key is reported as nonfinite (its bin is
  backend-dependent and never used).
- **Cells.** A multi-key library indexes cells row-major in the order of its axes: `(i0 * n1 + i1) * n2 + i2`
  (`Grid.ravel`).
- **Units and frames** are those of Offline: mm, ns, MeV, MeV/c, the Mu2e frame unless the axis says otherwise.
  Local frames (a bar, a component, an incident direction) are defined in the axis `frame` text and the library's
  own documentation, with the transformation from the Mu2e frame.
- **Outputs** are described the same way in the manifest `outputs`: name, unit, dtype, and the null answer.

## 3. Manifest

A manifest is a JSON object, stored in the library file (section 4), with every field below present. A field whose
value is not known, typically for a library built before this scheme, is `null` and is listed in `unrecorded` with
the reason. A manifest with nothing unrecorded is **complete**. A new library must be complete; a migrated one
carries its gaps openly and never invents provenance.

| Field | Content |
|---|---|
| `schema` | manifest schema, 1 |
| `name` | `<subsystem>.<what>.<configuration>`, e.g. `tracker.nrow.run1a` |
| `subsystem` | `core`, `crv`, `calo`, `tracker`, `flash`, `secondaries`, `beam`, `trigger` or `mixing` |
| `version` | positive integer, raised on every change of content |
| `kind` | `binned` (per-cell values or distributions), `pool` (per-cell donors drawn whole), `parametric` (fitted constants), `interpolation` (a curve or grid interpolated), `replay` (whole recorded interactions placed and replayed) |
| `description` | one paragraph: what question the library answers |
| `axes` | the keys, as `Axis.describe()`: name, unit, frame, array, scale, policy, nbins, lo, hi |
| `outputs` | what a lookup returns: name, unit, dtype, description (with the null answer) |
| `arrays` | every stored array: dtype and shape |
| `content_hash` | sha256 over the arrays (section 4) |
| `source` | the source samples: dataset names, selections, event counts |
| `generator` | what produced them: program, release, physics list, configuration |
| `release` | the code that built the library: repository, commit, entry point |
| `seeds` | every seed of the source samples and of the build |
| `geometry` | geometry or configuration identity: tag and hash |
| `config_hash` | digest of the build configuration, when one exists (the tracker response configurations' `recoresp.config.digest`) |
| `split` | training and held-out samples and the rule that keeps them disjoint |
| `dtype` | storage dtypes, the compute dtypes supported (float32 and float64) |
| `rng` | host and device conventions, streams, draws per row |
| `domain` | the covered domain in words and the declared fallbacks (name, levels) |
| `validation` | the status (section 6) and where the validation record is |
| `unrecorded` | `{field: reason}` for every null field |
| `legacy` (optional) | the format the library had before its manifest (its own schema number) |
| `interpolation` (optional) | how a lookup combines cells: `{method: nearest \| linear, axes: [...]}`; fixed for a validation |
| `assumptions` (optional) | modelling assumptions the library makes beyond its keys, each `{name, description, tested}` |

`manifest.create` builds and checks one; `manifest.validate` lists every problem at once.

## 4. Storage and distribution

- **Format.** A library file is a NumPy `.npz`, read with `allow_pickle=False`: plain arrays, no objects. The
  manifest is the array `__manifest__`, a string holding its canonical JSON. A frozen file that predates the scheme
  (a shipped file whose bytes are already pinned) keeps its bytes and carries the same manifest as a sidecar,
  `<stem>.manifest.json`; a file with both must hold the same manifest in each.
- **Other containers.** Libraries that live in another container (ROOT trees, the Offline binary lookup tables on
  CVMFS) carry a sidecar manifest in the repository, and their loader calls `store.verify(manifest, arrays)` on the
  arrays it decoded: the same checks as `load`.
- **Hashes** (`hashing`), one definition each:
  - `file_sha256`: sha256 of the file bytes. This is what a registry entry and a release asset pin.
  - `content_hash`: sha256 over the arrays, with names sorted; each array contributes its name, dtype (byte order
    included), shape and C-order bytes, length-prefixed. Two files with the same content hash hold the same arrays
    bit for bit, whatever their compression or container.
  - `digest`: sha256 of the canonical JSON (`sort_keys`, no spaces, NaN refused), cut to 16 hex digits. This is the
    short identifier of what a library was built from.
- **Load checks** (`store.load`): a manifest is present and passes the schema; the arrays are exactly the declared
  ones, with the declared dtypes and shapes; the content hash matches; the stored axis edges agree with the
  manifest; every field of `expect` (name, version, a pinned content hash, the legacy schema) matches. Every problem
  is listed in one `LibraryError`; the arrays come back read-only.
- **Shipped names.** A small library ships in git under `<package>/data/` and is resolved by name (`NAMED`). Its
  `data/README.md` describes the inputs in words for readers. The manifest is the record that is checked.
- **Registry and release assets.** A large or per-configuration library is not in git. The package's registry JSON
  (`{"schema": 1, "entries": {id: entry}}`, or the bare `{id: entry}` layout of the tracker response registry)
  maps an identifier to `file`, `sha256`, `url`, and the manifest's `name`, `version`, `content_hash` and
  `config_hash`. `Registry.resolve(spec)` takes a shipped name, an existing path, an id, `name@version`, `name` (the
  highest version) or a unique hash prefix. The file is downloaded into the cache when missing, and its sha256 is
  checked on every resolution, cached or not; a mismatch is refused. Cache: `$MU2EFS_LIBRARY_CACHE` (or the
  package's own variable), else `~/.cache/mu2efastsim/libraries`. Release assets of new libraries go under the tag
  `lib-<name>-v<version>` (`registry.release_url`); the tracker response tables keep their `resp-<hash>` tags.

### Pinning in run configurations

Some libraries have test doubles or development copies with the production name: a synthetic `LookupTable_6000_0`, or
a copy of a table staged on a node. For these, the pin belongs to the **run configuration**, not to the library's name.

- **Pinned loads.** A production run configuration lists every library it uses with its content hash, in an asset set
  in the repository (for the CRV, `crvfastsim/configs/assets_<set>.json`, named by `RunConfig.asset_pins`; reco-r3 and
  the kit task use the same set). A chain that loads through the run configuration checks the content hash of what
  it decoded against that list and refuses a mismatch, or a library missing from the set. A container that is
  pinned byte for byte may use its file sha256 as the pin instead, as the CRV tier-1b asset sets do (`tier1b_assets.SETS`).
- **Ad hoc loads.** A load by path or name, as in development work or with a test double, is allowed. Its content hash
  is still computed and recorded, with the status `unpinned`.
- **Provenance.** Every load is entered in `library.provenance` (`record`, `loaded`, `summary`, `all_pinned`) and
  carried on the loaded object. An output's provenance takes that record, so an output can never claim it ran on a
  pinned library when it did not.
- **Domain.** Out-of-domain and fallback counting is the same as for any other library.

This is the general rule for libraries in external containers, such as Offline binary tables or ROOT files on
CVMFS, and for anything else a test double with a production name can stand in for. Libraries in the repository's
own `.npz` format keep the stricter rule above: the manifest is checked on every load, and a file without one is
refused.

## 5. Domain

Every library declares its domain, and nothing out of it is answered silently:

- **Per key** (the axis `policy`):
  - `refuse`: the lookup raises `OutOfDomain` naming the rows, before drawing;
  - `clip`: the end bin answers, and this is declared;
  - `nan`: the output is NaN, and this is declared.
  No policy extrapolates.
- **Per cell** (`domain.fallbacks`): where the training sample is sparse, a declared fallback answers. Examples are
  a coarser cell, the nearest filled bin, or the whole sample. The library records which level served each cell
  (`cell_level`, `level`, `empty_bins`).
- **Counters.** `domain_counts(...)` returns a `DomainCounter`: lookups, values below, above and nonfinite per axis,
  rows served at each fallback level, and refusals. A device lookup returns its per-row codes
  (`axes.domain_codes_jax`) or levels next to its output, and the caller adds them after the device sync.
  `DomainCounter.record()` gives counts plus the **coverage** (1 minus the out-of-domain fraction, per axis) and the
  **out-of-domain fraction**. These two numbers go into the validation record and, for production outputs, into
  the output's provenance.
- **Capacity** (device buffers sized ahead) is not domain. An overflow is reported and the batch redone or refused,
  never truncated.

## 6. Validation and the common vocabulary

Each library carries a **validation record** (`fidelity.ValidationRecord`), named in its manifest. The record holds
the library identity (`Library.provenance()`), the criteria, the held-out sample, the results per dtype, any
reference comparisons, the performance entries and the status. The words below have one meaning across the
project, and they are the words used when a library is discussed.

### Status levels

One per library and dtype, strongest first; a library's status is the weaker of its two dtypes.

| Status | Meaning |
|---|---|
| **bit-identical** | the outputs equal a reference bit for bit, for the same inputs, keys and dtype (a migration, a refactor, a port) |
| **exact** | deterministic outputs equal a reference within a declared numerical tolerance (e.g. 1e-9 mm) |
| **statistically equivalent** | every criterion declared before the held-out read passes on the held-out sample, in float32 and in float64 |
| **tested, not closed** | measured, but a criterion fails, one dtype is missing, or a criterion was declared after the read |
| **not validated** | nothing measured yet |

"Statistically equivalent" covers what the criteria test, and no more. The record says which observables were
tested.

### Criteria are declared before the held-out read

A criterion (`fidelity.Criterion`) names the observable, the metric, the quantity compared and the rule:
`within_sigma` (|value − target| ≤ k·err + allowance), `abs_le`, `ge` or `le`. `ValidationRecord.declare` hashes it
and time-stamps it. `read_held_out` marks the first read of the held-out sample, and after that no criterion can be
added. A criterion chosen after looking is recorded in the notes, and the status stays **tested, not closed**. The
held-out sample is disjoint from the training sample by the manifest's `split` rule.

### Metrics

These are the definitions already used by the CRV and tracker validations, collected in `fidelity`:

| Metric | Definition | Use |
|---|---|---|
| **paired ratio ± error** | `paired_ratio(ref, cand)`: ratio of sums over units (events or shards) seen by both arms, with error `sqrt(n var(cand − R ref)) / sum(ref)`. Zero-activity units count | yields, efficiencies, rates (e.g. "stops per input 0.906 ± 0.037") |
| **paired shape** | `paired_shape(F, R)`: per tail-merged bin, the summed difference over shards, its paired error and pull; chi2/ndf with its null (n−1)/(n−3) | shapes measured on the same events |
| **shape chi2/ndf against a stock-vs-stock null** | `shape_test(cand, ref, ref_a, ref_b)`: two-sample chi2/ndf (ROOT "UU") of fast vs stock, next to the chi2/ndf of two stock realizations (e.g. two halves); `excess` = their ratio | shapes from independent samples; a value is read against its null, never against 1 alone |
| **realization spread** | `realization_spread(values)`: mean, standard deviation and error of the mean over seeds or donor sets | pool and library fluctuations; the noise floor of the other metrics |
| **rank correlation** | `rank_correlation(a, b)`: Spearman with average ranks | where only the ordering matters (design ranking) |
| **coverage and out-of-domain fractions** | `coverage(DomainCounter.record())` | how much of the held-out sample the library actually answered |

A comparison is **paired** when both arms see the same inputs and keys, and **held-out** when the inputs are
disjoint from the training sample. The agreement line of a result says both.

### Performance

Speed uses the bench vocabulary (`mu2efastsim.bench`, [bench.md](bench.md)). `fidelity.performance(bench_json,
stock_rate=..., match=...)` turns a bench report into the entry:

- **warm rate**: events/s over passes 2 and 3, device synchronised;
- **setup**: pass 1 minus the mean warm pass, which includes the library load;
- **N10** = 9 × setup × warm rate, the job size at which setup is 10 % of the wall time;
- **× stock at the same boundary**: warm rate over stock events/s per single-core grid slot for the same input →
  output.

When the stock number is not at the same boundary, the entry carries the label instead of a bare multiplier:
**stage** (a sub-stage only), **proxy** (another sample's timing at the same stages) or **partial** (some stages
missing). For example, `stage (38x)`. Records in this repository carry no billing units, and `check_public` refuses
them.

## 7. Libraries today and their migration

Each existing library moves onto the core in **its own pull request**, with one requirement: **its outputs are
bit-identical before and after, in float32 and float64**. A frozen copy of the pre-migration functions in the test
is compared bit for bit with the migrated module, on the shipped library. Behaviour changes (a counter that now
reports, a policy that should refuse) come in later pull requests and are validated as such.

| Library (manifest name) | Package, file | Keys and binning | Format before | Provenance, hash before | Out of domain before | Migration |
|---|---|---|---|---|---|---|
| `tracker.nrow.run1a` | `trkfastsim.recoresp.nrow`, `data/nrow_run1a.npz` | \|p\| × tan dip × r_max of the entry helix (38 × 24 × 36), pools of rows | npz, schema 1 | `reference` text; none | clip; fallback levels in `cell_level` | **done**: sidecar manifest (bytes unchanged), checked `load`, `Axis`/`locate`, `domain_counts`; bit-identical host and device, float32 and float64 |
| `calo.digi_response.run1a` | `calofastsim.param_response`, `data/digi_response_run1a.npz` | visible E (60 log bins) × T − window end (2 ns), pools per (event, crystal) | npz, schema 2 | `reference` text; none | clip; `empty_bins` borrow; survival 1 below | **done** (#177): sidecar, checked `load`, `domain_counts`; data README schema fixed |
| `tracker.albedo.run1a` | `trkfastsim.sim.albedo`, `data/albedo_run1a_v1.npz` | exit radius × cos × sign(u_r), 48 bins | npz, version in file name | none; edges checked against constants | clip | **done** (#174): sidecar, checked `Library.load`, `domain_counts` |
| `tracker.response.*` | `trkfastsim.recoresp.table`, `data/ce_loophelix_run1a*.npz`, `registry.json` | nrow (18) × tan dip (11), 129-node inverse CDFs, pools | npz, no schema | `config_hash` (16 hex) for registered tables; file sha256 in the registry; shipped tables none | clip; build-time widening `level` | **done** (#176): sidecars for the three shipped tables, checked `load`; registered release assets without a manifest load by their registry sha256 |
| `calo.showers.*` | `calofastsim.showers.library` | entry E × cos × disk radius | npz with JSON `metadata` | metadata | clip + `outside` flag, nearest-bin fallback | **in review**: manifest embedded by `save`; the registered pre-manifest CAL2 library (`showers/registered.json`) loads as before; `domain_counts` |
| `core.ebrem.g4emlow861`, `core.mott.g4_11_3_2` | `mu2efastsim.ebrem`, `emsc`, `data/*.npz` | Z × ln T × κ; Mott coefficients | npz | source-file sha256 inside | clamp; T > 1 GeV raises | **done** (#175): sidecars, checked loads, `convert` embeds the manifest |
| `core.cry.sea_level` | `mu2efastsim.cry_tables` | CRY tables | text | file sha256 pinned | range-checked | manifest via `verify` |
| `beam.flash_attenuation.al_xcom` | `beamfastsim.flash_attenuation`, `data/flash_al_xcom.csv` | photon energy (XCOM grid) | CSV | source URL in header | NaN outside, never clamps | **done** (#178): sidecar over the CSV arrays via `verify`; policy `nan` declared |
| `beam.flash_param.*` | `beamfastsim.flash_param` (branch `flash-param`) | none: 4 constants (c0, c1, σ, λ) | JSON | none | domain in docs | `kind: parametric` npz once a constants file ships |
| `crv.lookup.v6_*` | `crvfastsim.response.lookup` (Offline v6 tables on CVMFS) | scint x,y,z; Cerenkov β, θ, φ, r, z | Offline binary | sha256 in the run's asset manifest | bin −1: no photons, not counted | **done** (this change): pinned by the run configurations (`configs/assets_crvconditions_v6_0.json`, section 4 "Pinning in run configurations"); ad hoc loads recorded unpinned; `domain_counts` counts the stock −1 bins |
| `crv.sipm.photon_map` | `crvfastsim.response.sipm` | 40 × 40 pixels | ROOT TH2 | as above | n/a | **done** (this change): pinned with the lookup tables |
| `crv.tier1b.*` | `crvfastsim.tier1b`, `tier1b_assets` | group × momentum quantile × path-length quantile, context and sub-cells | ROOT trees + npz/JSON sidecars | pinned library sha256, sidecar sha256, code sha256 | fallbacks, `fit_inside`, counters in `info` | manifest per asset set (sidecar), `verify` on the decoded arrays; `info` counters → `DomainCounter` |
| `tracker.deltas.run1a` | `trkfastsim.sim.deltas` | step length bins | (not shipped yet) | — | — | native when it ships |
| `secondaries.component.*` | `mu2efastsim.component_response` (LIB2) | exact entrance states; trials | in memory only | labels only | `UnsupportedEntrance` raises | native: see section 8 |
| `flash.response.*` | (being designed) | species × E × start z, r in the source volume × cos θ × u_r | — | — | — | first native user (agreed): section 8 |

## 8. Native users

### Flash response library (first native user)

The flash response is a photon source × Geant4 response library. Photons start inside a source volume around the
stopping target and are transported through the stock detector, and the library tabulates the expected tracker
ionising deposit per photon. The layout below is agreed with the flash study, and its pilot shaped it: the deposit
probability per photon is small (order 1e-4), the response is strongly forward (cos θ > 0.95 carries about half of
the energy), it rises with photon energy and is nearly flat in z. Under this core it is:

- **name**: `flash.response.<configuration>`; **kind**: `binned`; **lookup**: a deterministic expected value with no
  draws. It must be jit-able on device arrays, vectorised over 1e5–1e6 photons, and run in float32 and float64. The
  interpolation is declared in the manifest (`interpolation`: nearest cell, or linear in log E) and stays fixed for a
  validation.
- **axes** (Mu2e frame), with the DS axis as the symmetry axis:
  - `species`: categorical;
  - `energy`: MeV, log;
  - `z`, `r`: the photon's **start position in the source volume** (a cylinder r ≤ 250 mm, z 5300–6450 mm), not a
    surface;
  - `cos_theta`: the direction cosine to +z;
  - `u_r`: the radial direction cosine about the DS axis, (x·ux + y·uy) / (r |u|). The tracker is an annulus, so
    outward and inward photons at the same (r, cos θ) differ.
- **assumptions** (manifest `assumptions`, each tested in the validation):
  - azimuthal symmetry about the DS axis (neither the field nor the hardware is exactly symmetric);
  - the target foils present inside the source volume, a fixed configuration with its own geometry tag.
- **outputs**: per cell, the mean deposit per photon (MeV), its variance per photon, the number of photons N, and
  the **number of photons with a nonzero deposit**. With deposits this rare, a variance resting on a handful of
  nonzero entries is not usable, and the count is what lets a user flag it. A time-window split (deposits after
  250 ns) may be added later as further outputs; the current objective has no time cut.
- **domain**:
  - photons only until e± are tabulated: other species are refused, never answered with zero;
  - energy below the lowest edge (0.05 MeV) answers 0 as a declared fallback, counted. The response just above that
    edge is not zero, so the fallback stays visible;
  - energy above the top edge (the campaign covers the hard brems tail, about 60–100 MeV) is refused and counted;
  - start positions outside the source volume are refused.
- **manifest**:
  - `source`: the response campaign. It has two direction classes, each with its own gun configuration (a forward
    cone cos θ 0.8–1, and the rest of the sphere), log-flat in E and uniform in the volume. The **sampling density
    per class** is recorded, so the cell weights can be reproduced;
  - `seeds`: per-job seeds;
  - `geometry`: tag and hash of the target configuration inside the volume and of the hardware outside it;
  - `split`: odd/even jobs, for the realization spread.
- **residual term**: the deposit not carried by photons from the target, measured from the DS-start inclusive
  reference, is a separate library (`flash.residual.<configuration>`, kind `parametric`) with its own uncertainty
  and provenance. It is not part of the photon table.
- **validation**: criteria declared before the held-out read, in float32 and float64:
  - Spearman rank correlation ≥ 0.9 on the stock design points;
  - absolute RMS ≤ 5 %;
  - budget-decision agreement ≥ 95 %;
  - then the same frozen library on held-out new designs.

  A design the chain refuses (for example a capped positron) is recorded as a failure, never as a zero. Speed is
  given at the per-design boundary and labelled **stage** until the whole objective is covered.

### Geometry-portable interaction banks (secondaries)

`component_response.Library` holds complete local trials per component. It moves onto the core as follows:

- **kind**: `replay`;
- **storage**: one `.npz`, with the column tables flattened under prefixes (`entrances_*`, `particles_*`, `steps_*`,
  `exits_*`) plus the three offset arrays; the constructor's checks run after `store.load`;
- **geometry**: the component's construction label and a hash of its construction and materials;
- **generator**: the physics label, plus the physics-list, material and cut hashes the capture records;
- **domain**: `UnsupportedEntrance` becomes the `refuse` policy, raised before any draw and counted by
  `DomainCounter`;
- **manifest completeness**: the bank is admissible only when its manifest is complete. The capture's open items
  (physics-list, material and cut hashes, energy accounting) are exactly its `unrecorded` fields until they close;
- **validation**: nominal and moved or rotated reference modules, with the vocabulary above.
