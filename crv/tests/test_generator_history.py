"""Synthetic scheduler and worker evidence guards."""
import pytest

from valhist import generator_history as H


def log(proc=0):
    return (f'=== grid_crygen: shard {proc} subrun {2001+proc} seed {20263000+proc} nevts 1000000 host node\n'
            'TrigReport Events total = 1000000 passed = 1000000 failed = 0\n'
            f'=== grid_crygen: done shard {proc}\ngrid_crygen.sh COMPLETED with exit status 0\n')


def ads():
    return [dict(ClusterId=H.CLUSTER, Owner='ram2aq', ProcId=s-2001, JobStatus=3,
                 Environment=f'NEVTS=1000000 BASESEED=20263000 SUBRUN0=2001 MUSING=MDC2025ay PROCESS={s-2001}',
                 LastHoldReason=' SYSTEM_PERIODIC_HOLD  Run Time/limit 10853/1.080000000000000E+04',
                 RemoveReason='via condor_rm (by user ram2aq)', NumJobStarts=1, RemoteWallClockTime=10853.,
                 RemoteUserCpu=1300., RemoteSysCpu=30., LastRemoteHost='slot@node') for s in sorted(H.B.MISSING)]


def test_complete_worker_and_six_removals():
    assert H.worker(log(), 0)['logged_events'] == 1000000
    rows = H.history(ads())
    assert len(rows) == 6 and all(r['cpu_wall_fraction'] < .13 for r in rows)


@pytest.mark.parametrize('kind', ['identity', 'partial', 'failed', 'duplicate'])
def test_worker_rejects(kind):
    text = log()
    text = {'identity': text.replace('subrun 2001', 'subrun 2002'),
            'partial': text.replace('total = 1000000', 'total = 999999'),
            'failed': text.replace('status 0', 'status 1'), 'duplicate': text + text}[kind]
    with pytest.raises(ValueError):
        H.worker(text, 0)


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'reason', 'starts', 'environment', 'status'])
def test_history_rejects(kind):
    records = ads()
    if kind == 'missing':
        records.pop()
    elif kind == 'duplicate':
        records.append(records[0])
    else:
        key, value = {'reason': ('LastHoldReason', 'memory'), 'starts': ('NumJobStarts', 2),
                      'environment': ('Environment', 'NEVTS=1'), 'status': ('JobStatus', 5)}[kind]
        records[0][key] = value
    with pytest.raises(ValueError):
        H.history(records)
