"""TMVA MLP weights (the XML files Offline's MVATools reads) evaluated in JAX, as MVATools::evalMVA does.

Inputs are normalised with the <Variables> Min/Max (float), hidden layers use the file's activation, and the
output is the logistic of the last linear sum (MVATools for TMVA releases >= 262657).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import jax.numpy as jnp
import numpy as np

F32 = jnp.float32


def load(path):
    root = ET.parse(path).getroot()
    opts = {o.get("name"): (o.text or "").strip() for o in root.iter("Option")}
    release = next((i.get("value") for i in root.iter("Info") if i.get("name") == "TMVA Release"), "[0]")
    if int(release.split("[")[1].split("]")[0]) < 262657:
        raise ValueError("pre-262657 TMVA networks (raw output) are not supported")
    act = opts.get("NeuronType", "sigmoid")
    if act not in ("sigmoid", "tanh"):
        raise ValueError(f"unsupported activation {act}")
    norm = "N" in opts.get("VarTransform", "")
    vmin = np.array([np.float32(v.get("Min")) for v in root.find("Variables")], np.float32)
    vmax = np.array([np.float32(v.get("Max")) for v in root.find("Variables")], np.float32)
    layers = []
    for layer in root.find("Weights").find("Layout"):
        rows = [np.array((n.text or "").split(), np.float64) for n in layer.findall("Neuron")]
        if rows and len(rows[0]):
            layers.append(np.stack(rows).astype(np.float32))       # [n_source incl. bias, n_target]
    return {"act": act, "norm": norm, "offset": vmin, "scale": (np.float32(2.0) / (vmax - vmin)).astype(np.float32),
            "layers": layers}


def evaluate(net, x):
    """x [..., nvar] (float) -> MVA output [...] in float32."""
    x = x.astype(F32)
    if net["norm"]:
        x = (x - net["offset"]) * net["scale"] - F32(1.0)
    one = jnp.ones(x.shape[:-1] + (1,), F32)
    for k, w in enumerate(net["layers"]):
        y = jnp.concatenate([x, one], axis=-1) @ jnp.asarray(w)
        if k < len(net["layers"]) - 1:
            y = F32(1.0) / (F32(1.0) + jnp.exp(-y)) if net["act"] == "sigmoid" else jnp.tanh(y)
        x = y
    return (F32(1.0) / (F32(1.0) + jnp.exp(-x[..., 0]))).astype(F32)
