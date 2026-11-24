"""TMVA MLP weights (the XML files Offline's MVATools reads) evaluated in JAX, as MVATools::evalMVA does.

Inputs are normalised with the <Variables> Min/Max (float), hidden layers use the file's activation, and the
output is the logistic of the last linear sum (MVATools for TMVA releases >= 262657). The file is read as MVATools
reads it: every number with strtof (correctly rounded to float), scale = 2.0 / (max - min) in double, stored as float.

`evaluate` is the fast matmul form; `evaluate_stock` is MVATools::evalMVA's arithmetic (sequential float sums in the
stock order, the double-literal normalisation, the sigmoid 1.0 / (1.0 + expf(-x)) in double with glibc's expf),
traced under `exact.strict`.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np

from .exact import g as _g

F32 = jnp.float32
F64 = jnp.float64


def strtof(s):
    """C strtof: the decimal string correctly rounded to float (nearest, ties to even)."""
    q = Fraction(s.strip())
    c = np.float32(float(q))
    best = None
    for v in (np.nextafter(c, np.float32(-np.inf)), c, np.nextafter(c, np.float32(np.inf))):
        key = (abs(Fraction(float(v)) - q), int(v.view(np.uint32)) & 1)
        if best is None or key < best[0]:
            best = (key, v)
    return np.float32(best[1])


def load(path):
    root = ET.parse(path).getroot()
    opts = {o.get("name"): (o.text or "").strip() for o in root.iter("Option")}
    release = next((i.get("value") for i in root.iter("Info") if i.get("name") == "TMVA Release"), "[0]")
    if int(release.split("[")[1].split("]")[0]) < 262657:
        raise ValueError("pre-262657 TMVA networks (raw output) are not supported")
    act = opts.get("NeuronType", "sigmoid")
    act = "relu" if act == "ReLU" else act
    if act not in ("sigmoid", "tanh", "relu"):
        raise ValueError(f"unsupported activation {act}")
    norm = "N" in opts.get("VarTransform", "")
    vmin = np.array([strtof(v.get("Min")) for v in root.find("Variables")], np.float32)
    vmax = np.array([strtof(v.get("Max")) for v in root.find("Variables")], np.float32)
    scale = np.array([np.float32(2.0 / float(d)) for d in (vmax - vmin)], np.float32)   # vscale_: double, stored float
    layers = []
    for layer in root.find("Weights").find("Layout"):
        rows = [np.array([strtof(t) for t in (n.text or "").split()], np.float32) for n in layer.findall("Neuron")]
        if rows and len(rows[0]):
            layers.append(np.stack(rows).astype(np.float32))       # [n_source incl. bias, n_target]
    return {"act": act, "norm": norm, "offset": vmin, "scale": scale, "layers": layers}


def evaluate(net, x):
    """x [..., nvar] (float) -> MVA output [...] in float32."""
    x = x.astype(F32)
    if net["norm"]:
        x = (x - net["offset"]) * net["scale"] - F32(1.0)
    one = jnp.ones(x.shape[:-1] + (1,), F32)
    for k, w in enumerate(net["layers"]):
        y = jnp.matmul(jnp.concatenate([x, one], axis=-1), jnp.asarray(w), precision="highest")   # float32 as stock, no TF32
        if k < len(net["layers"]) - 1:
            y = (F32(1.0) / (F32(1.0) + jnp.exp(-y)) if net["act"] == "sigmoid"
                 else jnp.maximum(F32(0.0), y) if net["act"] == "relu" else jnp.tanh(y))
        x = y
    return (F32(1.0) / (F32(1.0) + jnp.exp(-x[..., 0]))).astype(F32)


# --- MVATools::evalMVA, stock arithmetic ----------------------------------------------------------------------------

# glibc 2.28+ expf (sysdeps/ieee754/flt-32/e_expf.c, e_exp2f_data.c): 2^(k/32) table and the scaled polynomial
_EXP2F_T = np.array([
    0x3ff0000000000000, 0x3fefd9b0d3158574, 0x3fefb5586cf9890f, 0x3fef9301d0125b51,
    0x3fef72b83c7d517b, 0x3fef54873168b9aa, 0x3fef387a6e756238, 0x3fef1e9df51fdee1,
    0x3fef06fe0a31b715, 0x3feef1a7373aa9cb, 0x3feedea64c123422, 0x3feece086061892d,
    0x3feebfdad5362a27, 0x3feeb42b569d4f82, 0x3feeab07dd485429, 0x3feea47eb03a5585,
    0x3feea09e667f3bcd, 0x3fee9f75e8ec5f74, 0x3feea11473eb0187, 0x3feea589994cce13,
    0x3feeace5422aa0db, 0x3feeb737b0cdc5e5, 0x3feec49182a3f090, 0x3feed503b23e255d,
    0x3feee89f995ad3ad, 0x3feeff76f2fb5e47, 0x3fef199bdd85529c, 0x3fef3720dcef9069,
    0x3fef5818dcfba487, 0x3fef7c97337b9b5f, 0x3fefa4afa2a490da, 0x3fefd0765b6e4540], np.uint64)
_N = 32
_INVLN2N = float.fromhex("0x1.71547652b82fep+0") * _N
_SHIFT = float.fromhex("0x1.8p+52")
_C = (float.fromhex("0x1.c6af84b912394p-5") / _N / _N / _N, float.fromhex("0x1.ebfce50fac4f3p-3") / _N / _N,
      float.fromhex("0x1.62e42ff0c52d6p-1") / _N)
_OFLOW = np.float32(float.fromhex("0x1.62e42ep6"))
_UFLOW = np.float32(-float.fromhex("0x1.9fe368p6"))


def expf(x):
    """glibc's expf (the generic C build: no fused multiply-add) of float x, in float64 arithmetic; under
    exact.strict (the products are guarded against contraction)."""
    x = jnp.asarray(x).astype(F32)
    xd = x.astype(F64)
    z = _g(xd * _INVLN2N)
    kd = _g(z + _SHIFT)
    ki = jax.lax.bitcast_convert_type(kd, jnp.uint64)
    kd = kd - _SHIFT
    r = z - kd
    t = jnp.asarray(_EXP2F_T)[(ki & np.uint64(_N - 1)).astype(jnp.int32)] + (ki << np.uint64(52 - 5))
    sc = jax.lax.bitcast_convert_type(t, F64)
    zz = _g(_C[0] * r) + _C[1]
    r2 = _g(r * r)
    y = _g(_C[2] * r) + 1.0
    y = _g(zz * r2) + y
    y = (y * sc).astype(F32)
    return jnp.where(x > _OFLOW, np.float32(np.inf), jnp.where(x < _UFLOW, np.float32(0.0), y))


def _sigmoid_stock(a):
    return (1.0 / (1.0 + expf(-a).astype(F64))).astype(F32)


def stock_layers(net):
    """The float32 weight matrices of a net [n_source incl. bias, n_target] (static numpy)."""
    if net["act"] != "sigmoid":
        raise ValueError("evaluate_stock supports sigmoid networks")
    return [np.asarray(w, np.float32) for w in net["layers"]]


def evaluate_stock(net, x, layers=None):
    """MVATools::evalMVA on x [..., nvar] (float) -> [...] float32, under exact.strict. `layers` (optional) replaces
    the net's weight matrices by arrays of the same shapes (a per-row choice between two nets)."""
    x = jnp.asarray(x).astype(F32)
    if net["norm"]:
        p = _g((x - jnp.asarray(net["offset"], F32)) * jnp.asarray(net["scale"], F32))
        x = (p.astype(F64) - 1.0).astype(F32)
    ws = stock_layers(net) if layers is None else layers
    xs = [x[..., i] for i in range(x.shape[-1])] + [jnp.ones(x.shape[:-1], F32)]
    for k, w in enumerate(ws):
        w = jnp.asarray(w, F32)
        y = jnp.zeros(x.shape[:-1] + (w.shape[-1],), F32)
        for i in range(w.shape[-2]):
            y = y + _g(w[..., i, :] * xs[i][..., None])
        y = _sigmoid_stock(y)
        xs = [y[..., j] for j in range(y.shape[-1])] + [jnp.ones(x.shape[:-1], F32)]
    return xs[0]
