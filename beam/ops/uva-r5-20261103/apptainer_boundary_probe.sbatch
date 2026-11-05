#!/bin/bash
#SBATCH -J r5_apptainer_boundary
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:12:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --no-requeue
set -eo pipefail
ulimit -c 0
exec /usr/bin/python3.11 -I -B /home/ram2aq/mu2e/fastsim_uva/r5_flash_20261103/apptainer_boundary_probe.py --apptainer /apps/software/standard/core/apptainer/1.4.5/bin/apptainer
