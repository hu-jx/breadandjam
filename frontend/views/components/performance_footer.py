from views.helpers import render as render_template

_METRICS = {
    "accuracy":   "72.8%",
    "robustness": "81.9%",
    "last_eval":  "2026-08-31",
}

def render():
    render_template("performance", **_METRICS)