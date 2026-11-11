"""The common fidelity and performance vocabulary of libraries (docs/libraries.md, "Validation").

Status levels (one per library and dtype, strongest first):

    bit-identical              the outputs equal a reference bit for bit (same inputs, keys and dtype)
    exact                      deterministic outputs equal a reference within a declared numerical tolerance
    statistically equivalent   every criterion declared before the held-out read passes, in float32 and float64
    tested, not closed         measured, but a criterion fails, a dtype is missing, or a criterion was declared late
    not validated              nothing measured yet

Metrics (definitions shared with the CRV and tracker validations; NumPy only):

    paired_ratio      ratio of sums on the same events / shards, delta-method standard error from the per-unit pairs
    paired_shape      per-bin summed difference over shards, paired error, pulls, chi2/ndf and its null (n-1)/(n-3)
    shape_chi2        two-sample homogeneity chi2 of unweighted histograms (ROOT Chi2Test "UU"), (chi2, ndf)
    shape_test        shape_chi2 of candidate vs reference beside the stock-vs-stock null (two stock realizations)
    realization_spread  mean, standard deviation and error of the mean over realizations (seeds)
    rank_correlation  Spearman rank correlation (average ranks for ties), where only the ordering matters
    coverage          in-domain and out-of-domain fractions (from a DomainCounter record)

Speed is reported with the bench vocabulary (mu2efastsim.bench, docs/bench.md): warm rate, setup, N10, and the
multiplier against stock at the same boundary, labelled "stage", "proxy" or "partial" when the stock number is not at
the same boundary. Billing units never appear in a record: :func:`check_public` refuses them.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from .hashing import digest

STATUSES = ("bit-identical", "exact", "statistically equivalent", "tested, not closed", "not validated")
MATCHES = ("yes", "stage", "proxy", "partial")
DTYPES = ("float32", "float64")
_PRIVATE = re.compile(r"(^|[^a-z])su([^a-z]|$)", re.I)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------------------------------------------- metrics
def paired_ratio(ref, cand):
    """cand / ref as a ratio of sums over paired units (events or shards seen by both arms), with the delta-method
    standard error sqrt(n var(cand - R ref)) / sum(ref). Zero-activity units count. The definition of
    crvfastsim.validation.paired_rates.paired_ratio."""
    a, b = np.asarray(ref, float), np.asarray(cand, float)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("paired arrays must be finite, same-sized one-dimensional arrays with at least two units")
    n = len(a)
    ta, tb = float(a.sum()), float(b.sum())
    out = dict(metric="paired_ratio", units=n, ref=ta, cand=tb, difference=tb - ta,
               difference_err=float(np.sqrt(n * np.var(b - a, ddof=1))))
    if ta <= 0:
        return dict(out, ratio=None, err=None, status="undefined_zero_reference")
    r = tb / ta
    return dict(out, ratio=r, err=float(np.sqrt(n * np.var(b - r * a, ddof=1)) / ta), status="ok")


def merge_tail(F, R, min_ref=25):
    """Merge histogram bins from the top down until each merged bin's reference sum is at least min_ref."""
    tot = R.sum(0)
    groups, cur, acc = [], [], 0.0
    for b in range(len(tot) - 1, -1, -1):
        cur.insert(0, b)
        acc += tot[b]
        if acc >= min_ref:
            groups.insert(0, cur)
            cur, acc = [], 0.0
    if cur:
        groups = [cur + groups[0]] + groups[1:] if groups else [cur]
    merge = lambda X: np.stack([X[:, g].sum(1) for g in groups], 1)
    return merge(F), merge(R), groups


def paired_shape(F, R, min_ref=25):
    """Paired shape of per-shard histograms F, R [shards, bins]: per (tail-merged) bin the summed difference, its
    paired per-shard error and pull; chi2/ndf over bins with a nonzero error and its null expectation (n-1)/(n-3) for
    t-distributed pulls of n shards. The definition of crvfastsim.validation.run1a_vseries.paired_shape."""
    F, R = np.asarray(F, float), np.asarray(R, float)
    F, R, groups = merge_tail(F, R, min_ref)
    n = len(F)
    d = F - R
    D = d.sum(0)
    se = np.sqrt(n * d.var(0, ddof=1)) if n > 1 else np.zeros(D.shape)
    ok = se > 0
    pull = np.where(ok, D / np.where(ok, se, 1), 0.0)
    return dict(metric="paired_shape", shards=n, bins=groups, pull=pull.tolist(),
                chi2_ndf=float((pull[ok] ** 2).mean()) if ok.any() else float("nan"), ndf=int(ok.sum()),
                chi2_ndf_null=(n - 1) / (n - 3) if n > 3 else float("nan"),
                max_abs_pull=float(np.abs(pull[ok]).max()) if ok.any() else float("nan"))


def shape_chi2(a, b):
    """Two-sample chi2 of unweighted histograms a, b (ROOT Chi2Test "UU"): sum over bins filled in either of
    (sqrt(B/A) a - sqrt(A/B) b)^2 / (a + b). Returns (chi2, ndf). The definition of
    crvfastsim.objective.crvresponse.chi2_shape."""
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    A, B = a.sum(), b.sum()
    if A <= 0 or B <= 0:
        return 0.0, 0
    m = (a + b) > 0
    num = (np.sqrt(B / A) * a[m] - np.sqrt(A / B) * b[m]) ** 2
    return float((num / (a[m] + b[m])).sum()), int(m.sum() - 1)


def shape_test(cand, ref, ref_a=None, ref_b=None):
    """Shape chi2/ndf of cand against ref, beside the stock-vs-stock null: shape_chi2 of two stock realizations
    (ref_a, ref_b; e.g. two halves of the stock sample). ``excess`` = chi2/ndf over the null's chi2/ndf."""
    c2, ndf = shape_chi2(cand, ref)
    out = dict(metric="shape_chi2", chi2=c2, ndf=ndf, chi2_ndf=c2 / ndf if ndf else float("nan"))
    if ref_a is not None and ref_b is not None:
        n2, nndf = shape_chi2(ref_a, ref_b)
        null = n2 / nndf if nndf else float("nan")
        out.update(null_chi2=n2, null_ndf=nndf, null_chi2_ndf=null,
                   excess=out["chi2_ndf"] / null if nndf and null > 0 else float("nan"))
    return out


def realization_spread(values):
    """Spread over realizations (one value per seed or donor set): n, mean, std (ddof 1), error of the mean, range."""
    v = np.asarray(values, float).ravel()
    if v.size < 2 or not np.isfinite(v).all():
        raise ValueError("realization_spread needs at least two finite values")
    s = float(v.std(ddof=1))
    return dict(metric="realization_spread", n=int(v.size), mean=float(v.mean()), std=s,
                sem=s / np.sqrt(v.size), min=float(v.min()), max=float(v.max()))


def _ranks(x):
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x), float)
    r[order] = np.arange(len(x), dtype=float)
    xs = x[order]
    starts = np.r_[0, np.flatnonzero(np.diff(xs)) + 1]
    ends = np.r_[starts[1:], len(x)]
    for s, e in zip(starts, ends):
        if e - s > 1:
            r[order[s:e]] = 0.5 * (s + e - 1)
    return r


def rank_correlation(a, b):
    """Spearman rank correlation of a and b (Pearson correlation of average ranks)."""
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.shape != b.shape or a.size < 3 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("rank_correlation needs two finite arrays of the same length (at least 3)")
    ra, rb = _ranks(a), _ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return dict(metric="rank_correlation", n=int(a.size), spearman=float((ra * rb).sum() / den) if den else float("nan"))


def coverage(domain_record):
    """Coverage entry from a DomainCounter.record(): per-axis coverage, worst out-of-domain fraction, fallbacks."""
    return dict(metric="coverage", lookups=domain_record["lookups"], coverage=domain_record["coverage"],
                out_of_domain_fraction=domain_record["out_of_domain_fraction"],
                fallback_fraction={k: v["fraction"] for k, v in domain_record["fallbacks"].items()},
                refused=domain_record["refused"])


def bit_identical(a, b):
    """True when two arrays (or mappings of arrays) have the same dtype, shape and bytes."""
    if isinstance(a, dict) or isinstance(b, dict):
        return set(a) == set(b) and all(bit_identical(a[k], b[k]) for k in a)
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


# ------------------------------------------------------------------------------------------------- criteria
@dataclass(frozen=True)
class Criterion:
    """An agreement criterion, declared before any held-out read.

    ``metric`` names the metric (above); ``rule`` is one of ``abs_le`` (|value - target| <= tol), ``within_sigma``
    (|value - target| <= k * err, plus ``tol`` as an allowance), ``ge`` / ``le`` (value >= / <= tol).
    ``quantity`` is the key of the metric result compared (``ratio``, ``chi2_ndf``, ``excess``, ``spearman``, ...).
    """
    name: str
    observable: str
    metric: str
    quantity: str
    rule: str
    target: float = 0.0
    tol: float = 0.0
    k: float = 2.0
    held_out: str = ""

    def __post_init__(self):
        if self.rule not in ("abs_le", "within_sigma", "ge", "le"):
            raise ValueError(f"criterion {self.name!r}: unknown rule {self.rule!r}")

    def as_dict(self):
        return dict(name=self.name, observable=self.observable, metric=self.metric, quantity=self.quantity,
                    rule=self.rule, target=self.target, tol=self.tol, k=self.k, held_out=self.held_out)

    def passes(self, result):
        v = result.get(self.quantity)
        if v is None or not np.isfinite(v):
            return False
        if self.rule == "abs_le":
            return abs(v - self.target) <= self.tol
        if self.rule == "within_sigma":
            e = result.get("err")
            return e is not None and abs(v - self.target) <= self.k * e + self.tol
        return v >= self.tol if self.rule == "ge" else v <= self.tol


@dataclass
class ValidationRecord:
    """The validation record a library carries: its identity, the criteria (declared, hashed and time-stamped
    before the held-out read), the results per dtype, and the status that follows from them."""
    library: dict                                    # Library.provenance() or {name, version, content_hash}
    criteria: list = field(default_factory=list)     # [{criterion..., declared_utc, criterion_hash}]
    held_out_read_utc: str = None
    held_out: str = ""
    results: dict = field(default_factory=dict)      # dtype -> [{criterion, result, passed}]
    references: dict = field(default_factory=dict)   # dtype -> {"bit_identical": bool, "exact": tol or None}
    performance: list = field(default_factory=list)
    notes: str = ""

    def declare(self, criterion, when=None):
        if self.held_out_read_utc is not None:
            raise ValueError("criteria must be declared before the held-out read; this one is late: "
                             "record it in notes and leave the status at 'tested, not closed'")
        d = criterion.as_dict()
        self.criteria.append(dict(d, declared_utc=when or utc_now(), criterion_hash=digest(d)))
        return self

    def read_held_out(self, sample, when=None):
        """Mark the first read of the held-out sample (criteria are frozen from here on)."""
        if not self.criteria:
            raise ValueError("declare at least one criterion before reading the held-out sample")
        self.held_out, self.held_out_read_utc = sample, when or utc_now()
        return self

    def add(self, dtype, name, result):
        if dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}")
        if self.held_out_read_utc is None:
            raise ValueError("call read_held_out before adding held-out results")
        c = next((c for c in self.criteria if c["name"] == name), None)
        if c is None:
            raise KeyError(f"criterion {name!r} was not declared")
        crit = Criterion(**{k: c[k] for k in ("name", "observable", "metric", "quantity", "rule", "target", "tol",
                                              "k", "held_out")})
        check_public(result)
        self.results.setdefault(dtype, []).append(dict(criterion=name, result=result, passed=crit.passes(result)))
        return self

    def reference(self, dtype, *, bit_identical=None, exact_tol=None):
        """Record a comparison with a reference output (migration, port): bit-identical, or exact within exact_tol."""
        self.references[dtype] = dict(bit_identical=bit_identical, exact_tol=exact_tol)
        return self

    def status(self, dtype=None):
        """Status of one dtype, or of the library (the weaker of the two dtypes; a missing dtype caps it)."""
        if dtype is None:
            levels = [self.status(d) for d in DTYPES]
            return max(levels, key=STATUSES.index)
        ref = self.references.get(dtype, {})
        if ref.get("bit_identical"):
            return "bit-identical"
        if ref.get("exact_tol") is not None:
            return "exact"
        res = self.results.get(dtype)
        if not res:
            return "not validated"
        names = {c["name"] for c in self.criteria}
        late = any(c["declared_utc"] > self.held_out_read_utc for c in self.criteria)
        if not late and {r["criterion"] for r in res} == names and all(r["passed"] for r in res):
            return "statistically equivalent"
        return "tested, not closed"

    def as_dict(self):
        out = dict(library=self.library, criteria=self.criteria, held_out=self.held_out,
                   held_out_read_utc=self.held_out_read_utc, results=self.results, references=self.references,
                   performance=self.performance, notes=self.notes,
                   status={d: self.status(d) for d in DTYPES}, overall=self.status())
        check_public(out)
        return out


# ------------------------------------------------------------------------------------------------- performance
def performance(bench, *, stock_rate, match, stock_row=""):
    """A performance entry from a mu2efastsim.bench report (dict): warm rate, setup, N10, device, dtype and the
    multiplier against the stock events/s per single-core grid slot at the same boundary. A ``match`` other than
    "yes" labels the multiplier ("stage (38x)") so it is never read as a same-boundary number."""
    if match not in MATCHES:
        raise ValueError(f"match must be one of {MATCHES}")
    x = bench["warm_rate"] / stock_rate
    mult = float(f"{x:.2g}")
    out = dict(workflow=bench.get("workflow"), boundary=bench.get("boundary"), device=bench.get("device"),
               dtype=bench.get("dtype"), inputs=bench.get("inputs"), warm_rate=bench["warm_rate"],
               setup_s=bench["setup_s"], n10=bench["n10"], stock_rate=stock_rate, stock_row=stock_row, match=match,
               multiplier=mult if match == "yes" else None,
               label=f"{mult:g}x" if match == "yes" else f"{match} ({mult:g}x)")
    check_public(out)
    return out


def check_public(obj, path="record"):
    """Refuse billing units in a record bound for a public repository (any key with an "su" token, at any depth)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if _PRIVATE.search(str(k)):
                raise ValueError(f"{path}.{k}: billing units belong to the private record only")
            check_public(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            check_public(v, f"{path}[{i}]")
    return obj
