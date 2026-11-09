#!/bin/bash
# CI tier of one suite of the tests workflow. Writes run=true|false and args to $GITHUB_OUTPUT.
#   weekly schedule (WEEKLY_CRON) or a manual run with weekly: the core suite runs only the @pytest.mark.weekly
#     validations (--runweekly -m weekly); every other job skips (no other suite has weekly tests, and pytest would
#     exit 5 with none selected);
#   nightly schedule, manual run with runslow: run with --runslow;
#   push: run the per-PR tier;
#   pull_request: run only if a changed path matches DEPS (or .github/), with --runslow if one matches a line of
#     <DIR>/ci/runslow_paths.txt (extended regular expressions starting with ^, relative to the suite directory).
# NOJAX=true marks the no-JAX twins, which never run the weekly tier.
set -eo pipefail
WEEKLY_CRON="11 3 * * 0"
run=true; args=""; weekly=false
case "${EVENT:-}" in
  schedule)
    if [ "${SCHEDULE:-}" = "$WEEKLY_CRON" ]; then weekly=true; else args=--runslow; fi ;;
  workflow_dispatch)
    if [ "${DISPATCH_WEEKLY:-false}" = true ]; then weekly=true
    elif [ "${DISPATCH_RUNSLOW:-true}" = true ]; then args=--runslow; fi ;;
  pull_request)
    if [ -n "${BASE_SHA:-}" ]; then
      changed=$(git diff --name-only "$BASE_SHA"...HEAD)
      list="${DIR:-.}/ci/runslow_paths.txt"
      if ! printf '%s\n' "$changed" | grep -E -q "${DEPS:-.}|^\.github/"; then
        run=false
        echo "no change under this suite or its dependencies (${DEPS}): skipping"
      elif [ -f "$list" ]; then
        prefix=""; [ "${DIR:-.}" = . ] || prefix="${DIR}/"
        slow=$(grep -v -E '^\s*(#|$)' "$list" | sed "s#^\^#^${prefix}#" | paste -sd'|' -)
        if [ -n "$slow" ] && printf '%s\n' "$changed" | grep -E -q "$slow"; then
          args=--runslow
          echo "changed files match $list: running the slow tier too"
        fi
      fi
    fi ;;
esac
if [ "$weekly" = true ]; then
  if [ "${SUITE:-}" = core ] && [ "${NOJAX:-false}" != true ]; then
    args="--runweekly -m weekly"
  else
    run=false
    echo "weekly tier: only the core suite has weekly validations: skipping"
  fi
fi
echo "run=$run args=${args:-per-PR}"
{ echo "run=$run"; echo "args=$args"; } >> "${GITHUB_OUTPUT:-/dev/null}"
