"""Material-call episodes and directional plane intersections from selected-lane traces."""
import numpy as np


def _plain(columns):
    return {k: v.tolist() if isinstance(v, (np.ndarray, np.generic)) else v for k, v in columns.items()}


def material_episodes(a, row):
    """Group consecutive same-material calls, retaining chronological sampler order."""
    take = np.flatnonzero(a['material_row'] == row)
    take = take[np.lexsort((a['material_order'][take], a['material_step'][take]))]
    if not len(take):
        return dict(count=0)
    s, name = a['material_step'][take], a['material_name'][take]
    begin = np.r_[0, np.flatnonzero((name[1:] != name[:-1]) | (s[1:] > s[:-1] + 1)) + 1]
    end = np.r_[begin[1:] - 1, len(take) - 1]
    step_rows = np.flatnonzero(a['step_row'] == row)
    step_numbers = a['step_step'][step_rows]
    si = step_rows[np.searchsorted(step_numbers, s[begin])]
    ei = step_rows[np.searchsorted(step_numbers, s[end])]
    used = a['material_requested_mm'][take] * a['material_fraction'][take]
    return _plain(dict(count=len(begin), material=name[begin], first_step=s[begin], last_step=s[end],
                       calls=end - begin + 1, used_mm=np.add.reduceat(used, begin),
                       requested_mm=np.add.reduceat(a['material_requested_mm'][take], begin),
                       stopped=a['material_stopped'][take[end]],
                       p_before=np.linalg.norm(a['material_p_before'][take[begin]], axis=1),
                       p_after=np.linalg.norm(a['material_p_after'][take[end]], axis=1),
                       step_x0=a['step_x0'][si], step_x1=a['step_x1'][ei],
                       t0=a['step_t0'][si], t1=a['step_t1'][ei],
                       path0=a['step_path0'][si], path1=a['step_path1'][ei]))


def x_plane_samples(a, row, planes):
    """All directional chord crossings of supplied x planes; not VD hit matching."""
    take = np.flatnonzero(a['step_row'] == row)
    x0, x1 = a['step_x0'][take], a['step_x1'][take]
    planes = np.asarray(planes)
    delta = x1[:, 0] - x0[:, 0]
    fraction = np.divide(planes[None, :] - x0[:, :1], delta[:, None],
                         out=np.full((len(take), len(planes)), np.nan), where=delta[:, None] != 0)
    hit, plane = np.nonzero((fraction > 0) & (fraction <= 1))
    order = np.lexsort((plane, fraction[hit, plane], hit))
    hit, plane = hit[order], plane[order]
    f, selected = fraction[hit, plane], take[hit]
    x = x0[hit] + f[:, None] * (x1[hit] - x0[hit])
    return _plain(dict(plane=plane, step=a['step_step'][selected], direction=np.sign(delta[hit]).astype(int),
                       fraction=f, x=x, t=a['step_t0'][selected] + f * (a['step_t1'][selected] - a['step_t0'][selected]),
                       p_before=np.linalg.norm(a['step_p0'][selected], axis=1),
                       p_after=np.linalg.norm(a['step_p1'][selected], axis=1)))
