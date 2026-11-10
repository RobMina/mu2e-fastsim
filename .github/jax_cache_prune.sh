#!/bin/bash
# Keep one JAX compilation cache per suite and tier. The six suites share the repository's 10 GB cache budget and every
# run on main saves a new per-commit key, so without pruning GitHub evicts the least recently used entries, which are
# the nightly's slow-tier caches. After a save on main this deletes the older caches of the same suite and tier (any
# month and requirements hash) and the old keys without a tier; on other runs it only lists what it would delete.
#   usage: jax_cache_prune.sh SUITE TIER KEEP_KEY delete|list
set -eo pipefail
suite=$1; tier=$2; keep=$3; mode=$4
re="^jax-${suite}-[0-9]{4}-[0-9]{2}-[0-9a-f]{64}-(${tier}-)?[0-9a-f]{40}\$"
# only caches created before KEEP_KEY, so a run never deletes the cache of a newer commit saved meanwhile
all=$(gh cache list -R "$GITHUB_REPOSITORY" --key "jax-${suite}-" --limit 200 --json key,createdAt \
      --jq '.[]|"\(.createdAt) \(.key)"')
since=$(printf '%s\n' "$all" | awk -v k="$keep" '$2 == k {print $1}')
if [ -z "$since" ]; then echo "cache $keep not found: nothing pruned"; exit 0; fi
old=$(printf '%s\n' "$all" | awk -v s="$since" '$1 < s {print $2}' | grep -E "$re" || true)
if [ -z "$old" ]; then echo "no older ${suite}/${tier} caches"; exit 0; fi
while read -r k; do
  if [ "$mode" = delete ]; then gh cache delete -R "$GITHUB_REPOSITORY" "$k" && echo "deleted $k"
  else echo "would delete $k"; fi
done <<< "$old"
