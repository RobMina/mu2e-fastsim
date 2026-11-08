#!/bin/bash
# CI tier of one suite of the tests workflow. Writes run=true|false and args=--runslow|"" to $GITHUB_OUTPUT.
#   schedule, workflow_dispatch (runslow): run with --runslow; push: run the per-PR tier;
#   pull_request: run only if a changed path matches DEPS (or the workflow itself), with --runslow if one matches SLOW.
# A submodule pin bump shows up as a change of the submodule's path ("core", "crv", "tracker").
set -eo pipefail
run=true; args=""
case "${EVENT:-}" in
  schedule) args=--runslow ;;
  workflow_dispatch) [ "${DISPATCH_RUNSLOW:-true}" = true ] && args=--runslow ;;
  pull_request)
    if [ -n "${BASE_SHA:-}" ]; then
      changed=$(git diff --name-only "$BASE_SHA"...HEAD)
      if ! printf '%s\n' "$changed" | grep -E -q "${DEPS:-.}|^\.github/"; then
        run=false
        echo "no change under this suite or its dependencies (${DEPS}): skipping"
      elif [ -n "${SLOW:-}" ] && printf '%s\n' "$changed" | grep -E -q "$SLOW"; then
        args=--runslow
        echo "changed files match the slow-only contracts (${SLOW}): running the slow tier too"
      fi
    fi ;;
esac
echo "run=$run args=${args:-per-PR}"
{ echo "run=$run"; echo "args=$args"; } >> "${GITHUB_OUTPUT:-/dev/null}"
