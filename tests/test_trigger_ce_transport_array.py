"""Array tasks receive only their own partition, not inherited whole-export inputs."""
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_array_clears_whole_export_inputs(tmp_path):
    code = tmp_path / 'code'
    (code / 'trigger').mkdir(parents=True)
    (code / 'trigger/ce_transport.sbatch').write_text(
        f'"{sys.executable}" -c "import json,os; print(json.dumps(dict(os.environ)))"\n')
    counts = tmp_path / 'counts.txt'
    counts.write_text('2811\n2793\n2777\n2802\n2869\n2850\n2814\n2786\n2770\n2772\n')
    bash_env = tmp_path / 'bash-env'
    bash_env.write_text('mapfile() { [[ $1 == -t && $2 == counts ]] || return 2; '
        'counts=(); while IFS= read -r row; do counts+=("$row"); done; }\n')
    env = dict(os.environ, BASH_ENV=str(bash_env), TRIGGER_ROOT=str(tmp_path), TRIGGER_CODE=str(code),
        TRIGGER_SPLIT=str(tmp_path / 'split'), TRIGGER_COUNT_FILE=str(counts),
        SLURM_ARRAY_TASK_ID='2', TRIGGER_EXTRA_TABLES='must-not-propagate.root',
        TRIGGER_SOURCE_MANIFEST='must-not-propagate.txt',
        TRIGGER_TRANSPORT_OUTPUT_ROOT=str(tmp_path / 'new-output'))
    result = subprocess.run(['bash', str(ROOT / 'trigger/ce_transport_array.sbatch')],
        env=env, capture_output=True, text=True, check=True)
    actual = json.loads(result.stdout)
    assert 'TRIGGER_EXTRA_TABLES' not in actual
    assert 'TRIGGER_SOURCE_MANIFEST' not in actual
    assert actual['TRIGGER_TABLES'] == str(tmp_path / 'split/parent_2.root')
    assert actual['TRIGGER_SOURCE'] == str(tmp_path / 'out/ce_regen_full/unfiltered_2.art')
    assert actual['TRIGGER_EXPECTED_EVENTS'] == '2793'
    assert actual['TRIGGER_OUTPUT'] == str(tmp_path / 'new-output/parent_2')
