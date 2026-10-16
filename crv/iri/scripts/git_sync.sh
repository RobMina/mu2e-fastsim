#!/bin/bash
# Clone or update the public crvfastsim repo on Eagle from a Polaris compute node (HTTPS through the ALCF proxy).
# usage (positional args via make_inline_job.py --args): git_sync.sh [fresh]   ("fresh" removes the clone first)
CODE=${CRVFS_EAGLE:?}/code
URL=https://github.com/RobMina/mu2e-crv-fastsim.git
export https_proxy=http://proxy.alcf.anl.gov:3128 http_proxy=http://proxy.alcf.anl.gov:3128
if [ "${1:-}" = fresh ] && [ -d "$CODE" ]; then
  rm -rf "$CODE" && echo "removed $CODE"
fi
if [ -d "$CODE/.git" ]; then
  git -C "$CODE" remote set-url origin "$URL"
  git -C "$CODE" pull --ff-only 2>&1 | tail -3
else
  git clone "$URL" "$CODE" 2>&1 | tail -3
fi
git -C "$CODE" remote -v | head -1
git -C "$CODE" log -1 --format='HEAD %h %ad %s' --date=iso 2>&1
ls "$CODE" 2>&1 | tr '\n' ' '; echo
echo "exit=0"
