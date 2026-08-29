from views.helpers import render as render_template

_METRICS = {
    "accuracy":   "1000% trust",
    "robustness": "100%",
    "speed":      "1 ms",
    "last_eval":  "2026-08-24",
}

def render():
    render_template("performance", **_METRICS)