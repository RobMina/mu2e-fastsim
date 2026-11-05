#!/bin/bash
#SBATCH -J r5_pidfd_self
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH --nodelist=udc-aw32-5c1
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -t 00:02:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --no-requeue
set -eo pipefail
ulimit -c 0
ulimit -f 64
exec /usr/bin/python3.11 -I -B /home/ram2aq/mu2e/fastsim_uva/r5_flash_20261103/pidfd_capability_probe.py
