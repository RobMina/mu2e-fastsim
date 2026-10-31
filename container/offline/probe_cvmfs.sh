#!/bin/bash
# Run inside an EL9 image on an allocated CPU node, with /dev/fuse and node-local scratch bound.
set -eo pipefail
OUT=${1:?usage: probe_cvmfs.sh OUTPUT NODE_LOCAL}
LOCAL=${2:?need node-local scratch}
mkdir -p "$OUT" "$LOCAL"
exec > >(tee "$OUT/probe.log") 2>&1
printf 'start=%s host=%s\n' "$(date -u +%FT%TZ)" "$(hostname)"
cat /etc/os-release
uname -a
ls -l /dev/fuse
cat /proc/sys/user/max_user_namespaces
[ ! -e /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh ] || { echo 'FAIL: host CVMFS leaked'; exit 2; }
for tool in unshare mount; do command -v "$tool"; done
unshare -Ur true
CE="$LOCAL/cvmfsexec"
cd "$CE"
cat > dist/etc/cvmfs/default.local <<EOF
CVMFS_HTTP_PROXY=DIRECT
CVMFS_CACHE_BASE=$LOCAL/cache
CVMFS_QUOTA_LIMIT=6000
CVMFS_TIMEOUT=30
CVMFS_TIMEOUT_DIRECT=30
EOF
mkdir -p "$LOCAL/cache" "$CE/dist/etc/cvmfs/domain.d"
cat > "$CE/dist/etc/cvmfs/domain.d/opensciencegrid.org.local" <<'EOF'
CVMFS_SERVER_URL="http://cvmfs-s1fnal.opensciencegrid.org:8000/cvmfs/@fqrn@;http://cvmfs-s1bnl.opensciencegrid.org:8000/cvmfs/@fqrn@"
EOF
for pass in cold warm; do
  "$LOCAL/time" -v -o "$OUT/$pass.time" ./cvmfsexec config-osg.opensciencegrid.org mu2e.opensciencegrid.org fermilab.opensciencegrid.org -- \
    /bin/bash -c '
      set -e
      export MU2E_OSNAME=al9 MU2E_SPACK=true UPS_OVERRIDE="-H Linux64bit+5.14-2.34-al9-3"
      source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh || true
      setup muse
      cd /cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025ba
      muse setup || echo "muse setup returned $?; verify executable and paths"
      export FHICL_FILE_PATH="$FHICL_FILE_PATH:$MUSE_WORK_DIR/build/$MUSE_STUB"
      printf "MUSE_WORK_DIR=%s MUSE_ENVSET=%s MUSE_STUB=%s\n" "$MUSE_WORK_DIR" "$MUSE_ENVSET" "$MUSE_STUB"
      command -v mu2e
      mu2e --version
      fhicl-dump -c Production/JobConfig/reco/Extracted.fcl
    ' > "$OUT/$pass.log" 2>&1 || { rc=$?; cat "$OUT/$pass.log"; cp -r "$CE/log" "$OUT/cvmfs-log-$pass" 2>/dev/null || true; exit "$rc"; }
  du -sh "$LOCAL/cache" | tee "$OUT/$pass.cache"
done
cp -r "$CE/log" "$OUT/cvmfs-log" 2>/dev/null || true
printf 'end=%s\n' "$(date -u +%FT%TZ)"
