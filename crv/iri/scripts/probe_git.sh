#!/bin/bash
# Probe GitHub ssh access from a Polaris compute node: direct, then through the ALCF proxy with each available tool.
echo "host=$(hostname)"
for t in nc ncat socat connect-proxy corkscrew; do printf "%-14s %s\n" "$t" "$(command -v $t || echo -)"; done
nc -h 2>&1 | head -2
try() {
  echo "--- $1"
  timeout 25 ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 \
    -i ~/.ssh/crvfastsim_deploy -o IdentitiesOnly=yes -p 443 "${@:2}" git@ssh.github.com 2>&1 | tail -2
}
try "direct" -o ProxyCommand=none
command -v nc    >/dev/null && try "nc -X connect"  -o "ProxyCommand=nc -X connect -x proxy.alcf.anl.gov:3128 %h %p"
command -v ncat  >/dev/null && try "ncat --proxy"   -o "ProxyCommand=ncat --proxy proxy.alcf.anl.gov:3128 --proxy-type http %h %p"
command -v socat >/dev/null && try "socat PROXY"    -o "ProxyCommand=socat - PROXY:proxy.alcf.anl.gov:%h:%p,proxyport=3128"
echo "--- https through proxy"
https_proxy=http://proxy.alcf.anl.gov:3128 timeout 20 git ls-remote https://github.com/RobMina/mu2e-xcards.git 2>&1 | head -2
echo exit=0
