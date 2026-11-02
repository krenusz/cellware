"""Minimal walker over rdata's raw parse tree, for Seurat objects rdata cannot convert."""

from __future__ import annotations

import numpy as np
import scipy.sparse


def _tag(t):
    while t is not None and not isinstance(t, (bytes, str)):
        t = t.referenced_object if t.info.type.name == "REF" else t.value
    return t.decode() if isinstance(t, bytes) else t


def attrs(o):
    """Attribute pairlist (S4 slots) -> {name: RObject}."""
    out, a = {}, o.attributes
    while a is not None and a.info.type.name == "LIST":
        out[_tag(a.tag)] = a.value[0]
        a = a.value[1] if len(a.value) > 1 else None
    return out


def names_of(o):
    a = attrs(o).get("names")
    return [_tag(v) for v in a.value] if a is not None else []


def vec(o):
    """Atomic vector or character vector -> numpy array."""
    if o.info.type.name == "STR":
        return np.array([_tag(v) if v.value is not None else "" for v in o.value], dtype=object)
    return np.asarray(o.value)


def factor(o):
    levels = [_tag(v) for v in attrs(o)["levels"].value]
    codes = np.asarray(o.value)
    return np.array([levels[c - 1] if c > 0 else "" for c in codes], dtype=object)


def column(df, name):
    """Column of an R data.frame (list with names) by name."""
    col = df.value[names_of(df).index(name)]
    return factor(col) if "levels" in attrs(col) else vec(col)


def dgc(o):
    s = attrs(o)
    shape = tuple(int(v) for v in vec(s["Dim"]))
    return scipy.sparse.csc_matrix((vec(s["x"]), vec(s["i"]), vec(s["p"])), shape)


def named_list_item(o, name):
    return o.value[names_of(o).index(name)]
