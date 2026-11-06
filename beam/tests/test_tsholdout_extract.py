import os
from pathlib import Path
import subprocess

import pytest

from beamfastsim import tsrawreference

SCRIPT = Path(tsrawreference.__file__).with_name('tsholdout_extract.sbatch')


@pytest.mark.parametrize('index,ok', [('99', False), ('100', True), ('199', True), ('200', False), ('0100', False)])
def test_only_heldout_explicit_indices_without_simulation(tmp_path, index, ok):
    log = tmp_path / 'calls'
    setup = tmp_path / 'setup.sh'
    setup.write_text('module() { :; }\napptainer() { printf "%s\\n" "$*" >> "$CALL_LOG"; }\n')
    out = tmp_path / 'out'
    out.mkdir()
    env = dict(os.environ, BASH_ENV=str(setup), CALL_LOG=str(log), TS_CAMPAIGN_ROOT='/existing/campaign',
               TS_GENERATION_JOB='22506112', TS_HELDOUT_OUT=str(out), TS_BEAM_CODE='/code/beam',
               TS_ANALYSIS_IMAGE='/existing/analysis.sif', TS_ANALYSIS_BIND='/existing,/code,' + str(tmp_path),
               SLURM_ARRAY_TASK_ID=index)
    result = subprocess.run(['bash', str(SCRIPT)], env=env, capture_output=True, text=True)
    assert (result.returncode == 0) == ok, result.stderr
    if ok:
        calls = log.read_text().splitlines()
        assert len(calls) == 2
        assert all('--bind ' + env['TS_ANALYSIS_BIND'] in call for call in calls)
        missing = dict(env)
        del missing['TS_ANALYSIS_BIND']
        rejected = subprocess.run(['bash', str(SCRIPT)], env=missing, capture_output=True, text=True)
        assert rejected.returncode != 0 and 'TS_ANALYSIS_BIND' in rejected.stderr
        assert log.read_text().splitlines() == calls
        assert '--cohort heldout' in calls[0]
        assert 'beamfastsim.tscampaign' in calls[0]
        assert 'beamfastsim.tsrawreference' in calls[1]
        assert f'/shards/{index}/tsrun1a_raw.root' in calls[1]
        (out / f'reference-{index}.npz').touch()
        again = subprocess.run(['bash', str(SCRIPT)], env=env, capture_output=True)
        assert again.returncode != 0
        assert len(log.read_text().splitlines()) == 2
    else:
        assert not log.exists()


def test_script_syntax_and_analysis_only():
    subprocess.run(['bash', '-n', str(SCRIPT)], check=True)
    text = SCRIPT.read_text()
    assert 'muse build' not in text
    assert 'tstransport' not in text
    assert '--array=100-199%16' in text
