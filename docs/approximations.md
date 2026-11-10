# Approximations

Each workflow replaces some stock stages with a model: a condensed transport, a library, a response layer. This page
lists them per workflow, with the switch that selects a level where one exists. Each entry also gives the
stock-equivalent alternative. Some approximations have no switch yet. Those are listed as "no switch" and cannot be
selected.

## Selecting levels

Every chain command and the shared timer take the same three options:

```bash
--approx-preset default|fast|fidelity      # all approximations at once
--approx NAME=LEVEL                        # one approximation, after the preset (repeatable)
--approximations FILE                      # a config with an approximations section (JSON, or YAML with PyYAML)
```

A config section names one workflow's levels, or several workflows keyed by name. With `--approx-preset`, the
preset on the command line wins over the file. `--approx` is applied last.

```yaml
approximations:
  W5: {preset: fidelity, pulse_fit: short_window}
  W2: {calo_albedo: none}
```

The presets:

| preset | each approximation at |
|---|---|
| `default` | its default level: what the chain does without these options |
| `fast` | its fastest level |
| `fidelity` | the level closest to stock that the code has; an approximation with no such level stays at its default and is listed under `no_fidelity_path` |

### What the levels do and do not change

- **Defaults.** A default level sets no option, so a run without the options passes exactly the arguments it did
  before. The test suites check that the default levels set nothing.
- **Levels map to existing options.** Each other level sets options the chain already has, for example
  `runchain --below-msc` or `cetracker --complete-outer-wall`.
- **Conflicts are refused.** An option given explicitly with a different value is refused. The exception is a level
  that refines an option the run chose, such as `response="default"` to `"pools"`. Such a level needs that option
  given and never turns an output on by itself.
- **Missing levels are refused.** A level an approximation does not have is refused, and so is an approximation that
  has no switch. Nothing is invented where no path exists.
- **Bench entry points.** The shared timer (`python3 -m mu2efastsim.bench`) finds the declarations in
  `<package>.approximations` from the entry point's package and `--workflow`, and adds the levels' arguments to
  `--kwargs`.
  - An entry point that does not take a level's argument refuses that level. For example, the W1 bench entry runs
    only the condensed chain.
- **Precision is separate.** Device precision (float32 or float64) is not a level here. The chains keep their own
  precision options.

## Provenance

Every run records an `approximations` entry (schema 1):

- `workflow`, `preset` and `overrides`.
- `levels`: what ran.
- `selected`: what was asked for. It differs from `levels` where the run's other options decided otherwise, for
  example no albedo on the NumPy tracker, or the full-length pulse fit without `--jax`. The reason is under
  `effective`.
- `non_default`.
- `no_fidelity_path` and `no_stock_path`.
- `fixed`: the approximations with no switch.

Where each workflow writes it:

| workflow | written to |
|---|---|
| W1 | `beamfastsim.tsdevice`: the NPZ `meta` |
| W2-W4 | `beamfastsim.cetracker`: `handoff.json` |
| W5 | `crvfastsim runchain`: `runchain_stats.json`, the `--emit` file and the `--save-reco` NPZ. `runchain-merge` refuses shards with different levels. `crvfastsim.analyses.runchain_bench`: its report |
| W6 | `mixing.stream_benchmark`: `configuration.json` |
| W7 | the timer report only (the device chain has no command) |
| W8 | `trigger.device.run decide`: the decide JSON |

Every workflow also writes the entry to the timer report.

To print the record a selection would write:

```bash
python3 -m mu2efastsim.approximations crvfastsim --record W5 --approx-preset fidelity
```

## The table

The table is generated from the declarations; a test keeps it in sync. To regenerate it:

```bash
python3 -m mu2efastsim.approximations beamfastsim crvfastsim calofastsim mixing trigger
```

The alternative column says what gives the stock result:

| alternative | meaning |
|---|---|
| exact port | a level of the chain runs the ported stock algorithm |
| stock art | stock products read from art or EventNtuple replace the approximated stage, through a different input of the chain or a stock job |
| none | the package has no stock-equivalent path yet |

<!-- table: python3 -m mu2efastsim.approximations -->
<!-- end table -->

## Notes per workflow

- **W1.** `ts_transport=stock_ds_entrance` runs the DS stage from the reference's stock DS-entrance states. This is
  `tsdevice --arm stock-ds`, and it is the only level of the beam stage. The step sizes and caps are numeric settings
  (`tsdevice --step`, `Settings`), not approximation levels.
- **W2-W4.** One declaration serves the stops (W2), DIO (W3) and RPC (W4) electrons of `cetracker`.
  - The albedo levels act on the device tracker; the NumPy tracker has none.
  - `straw_wall=complete` is the existing opt-in complete outer wall.
  - `response_draw=empirical` is the opt-in empirical joint draw (`DeviceChain(response="pools")`). It replaces
    `response="default"` and is refused when no response is requested. The fidelity preset leaves it at the copula
    there, and in `cetracker`, which applies no response layer. The record keeps the reason.
  - The stock route for the response layer and the primary-only steps is the stock reconstruction of the injector
    file. It is a separate job, not a level.
- **W5.**
  - `pulse_fit=full_length` runs the NumPy response, which fits every waveform at full length. It cannot be combined
    with `--jax`, and the runchain timer, which always runs on JAX, refuses it.
  - `below_msc` matters only for run configurations with a below-plane world.
  - The stock-art input for the transport and the secondaries is Geant4 CrvSteps through `--steps-input`.
- **W6, W7, W8.** These have no switchable approximation yet. The stock alternatives are the native art mixers (W6),
  stock steps through `make_steps_chain` (W7), and the stock trigger replay on the same digis (W8).
