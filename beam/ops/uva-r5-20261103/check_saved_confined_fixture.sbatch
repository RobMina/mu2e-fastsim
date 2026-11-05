#!/bin/bash
#SBATCH -J r5_confined_fixture_checks
#SBATCH -A ldmxuva
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:05:00
#SBATCH --nodelist=udc-aw32-7c1
#SBATCH --no-requeue
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
set -eo pipefail
ulimit -c 0
exec /usr/bin/python3.11 -I -B /home/ram2aq/mu2e/fastsim_uva/r5_flash_20261103/check_saved_confined_fixture.py
