"""Small helpers shared by the generated class reference pages."""

import inspect
from html import escape

import crabbymetrics as cm
import numpy as np
from IPython.display import HTML, display


def html_table(headers, rows):
    parts = ["<table>", "<thead>", "<tr>"]
    parts.extend(f"<th>{escape(str(h))}</th>" for h in headers)
    parts.extend(["</tr>", "</thead>", "<tbody>"])
    for row in rows:
        parts.append("<tr>")
        parts.extend(f"<td>{cell}</td>" for cell in row)
        parts.append("</tr>")
    parts.extend(["</tbody>", "</table>"])
    return "".join(parts)


def public_methods(cls):
    rows = []
    for name in sorted(n for n in dir(cls) if not n.startswith("_")):
        fn = getattr(cls, name)
        if callable(fn):
            rows.append([
                f"<code>{escape(name)}{escape(str(inspect.signature(fn)))}</code>"
            ])
    return rows


def summary_shape_rows(summary):
    return [
        [f"<code>{escape(str(k))}</code>", f"<code>{escape(str(np.shape(v)))}</code>"]
        for k, v in summary.items()
    ]
