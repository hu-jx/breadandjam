from views.helpers import render as render_template

def render():
    render_template(
        "empty_card",
        message="put confusion matrix and other stuff here",
    )