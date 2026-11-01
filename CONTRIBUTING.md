# Contributing

All changes to `main` of the four repositories (this umbrella, `mu2e-fastsim-core`, `mu2e-crv-fastsim`,
`mu2e-tracker-fastsim`) go through a pull request with passing tests and a review. Nobody pushes to `main` directly.

## Workflow

1. Branch from `main` (`git switch -c <topic> origin/main`); one topic per branch.
2. Keep defaults bit-identical unless the change is meant to move them; new behaviour is opt-in until validated.
3. Push the branch and open a pull request against `main`. Describe what changes, how it was validated (tests, numbers
   with uncertainties, the reference used), and what is not done.
4. The `tests` workflow runs on the pull request. A review and a green `tests` run are required before merging.
5. After a submodule's pull request merges, open an umbrella pull request that moves its pin; the umbrella's tests run
   the pinned commits together.

## Changes that span repositories

Use the **same branch name** in every repository involved. Each repository's CI checks out its siblings at that branch
when it exists there, otherwise at `main` (`core/ci/checkout_siblings.sh`). Merge in dependency order: core, then
crv, then tracker, then the umbrella pin.

## Tests

The CI environment is Ubuntu, Python 3.13, the packages pinned in `core/ci/requirements.txt` (CPU JAX), and the two
public Offline files fetched by `core/ci/offline_data.sh` on `MU2E_SEARCH_PATH`. No site configuration, no CVMFS and
no private data are available there, so:

- a test that needs site data or a GPU must skip cleanly when it is absent (`pytest.importorskip`, or a `skipif` on
  the missing variable or file), never fail;
- large statistical validations carry `@pytest.mark.slow` and run only with `--runslow`;
- `tests/test_no_site_paths.py` rejects user names and site paths in tracked files.

To reproduce CI locally from a directory holding the four checkouts side by side:

```bash
python3 -m pip install -r core/ci/requirements.txt
bash core/ci/offline_data.sh "$PWD/offline_data"
cd crv && MU2E_SEARCH_PATH=$PWD/../offline_data JAX_PLATFORMS=cpu PYTHONPATH=$PWD:$PWD/../core python3 -m pytest -q tests
```
