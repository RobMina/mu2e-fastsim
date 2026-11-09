# Contributing

All changes to `main` go through a pull request with passing tests and a review. Nobody pushes to `main` directly.

## Workflow

1. Branch from `main` (`git switch -c <topic> origin/main`); one topic per branch.
2. Keep defaults bit-identical unless the change is meant to move them; new behaviour is opt-in until validated.
3. Push the branch and open a pull request against `main`. Describe what changes, how it was validated (tests, numbers
   with uncertainties, the reference used), and what is not done.
   A change that moves a chain's validation result or its speed also updates that chain's row in
   [`docs/performance.md`](docs/performance.md): validity as summary statistics with errors (never raw histogram
   contents), throughput as events per second with the stock comparison defined there.
4. The `tests` workflow runs on the pull request: each suite whose directory or dependencies changed. A review and a
   green `tests` run are required before merging. A change that spans packages is one pull request.

## Tests

The CI environment is Ubuntu, Python 3.13, the packages pinned in `core/ci/requirements.txt` (CPU JAX), and the two
public Offline files fetched by `core/ci/offline_data.sh` on `MU2E_SEARCH_PATH`. No site configuration, no CVMFS and
no private data are available there, so:

- a test that needs site data or a GPU must skip cleanly when it is absent (`pytest.importorskip`, or a `skipif` on
  the missing variable or file), never fail;
- large statistical validations carry `@pytest.mark.slow` and run only with `--runslow`;
- `tests/test_no_site_paths.py` rejects user names and site paths in tracked files.
- CI runs every suite twice: with the pinned packages and without JAX (`pytest (no JAX)`, `no JAX (...)`), where every
  JAX test must skip cleanly (`pytest.importorskip("jax")`).

To reproduce one CI suite from the repository root:

```bash
python3 -m pip install -r core/ci/requirements.txt
bash core/ci/offline_data.sh "$PWD/offline_data"
cd crv && MU2E_SEARCH_PATH=$PWD/../offline_data JAX_PLATFORMS=cpu PYTHONPATH=$PWD:$PWD/../core python3 -m pytest -q tests
```
