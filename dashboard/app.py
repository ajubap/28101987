"""Entry point for the General CFD Surrogate Workbench dashboard.

Run with::

    python run_dashboard.py
    # or
    python -m dashboard.app

A multi-page Plotly Dash app. Each workflow step is its own page under
``dashboard/pages``. The active config is held in a session ``dcc.Store`` and
shared with every page via the navbar dropdown.
"""

from __future__ import annotations

from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

from dashboard import runner
from dashboard.theme import EXTERNAL_STYLESHEETS

app = dash.Dash(
    __name__,
    use_pages=True,
    pages_folder=str(Path(__file__).parent / "pages"),
    external_stylesheets=EXTERNAL_STYLESHEETS,
    suppress_callback_exceptions=True,
    title="CFD Surrogate Workbench",
)
server = app.server  # for gunicorn / deployment


def _nav_links():
    pages = sorted(dash.page_registry.values(), key=lambda p: p.get("order", 99))
    return [
        dbc.NavItem(dbc.NavLink(page["name"], href=page["relative_path"], active="exact"))
        for page in pages
    ]


def _config_selector():
    configs = runner.list_configs()
    default = configs[0] if configs else runner.DEFAULT_CONFIG
    return dbc.Row(
        [
            dbc.Col(html.I(className="bi bi-sliders text-white"), width="auto", className="pt-2"),
            dbc.Col(
                dcc.Dropdown(
                    id="config-dropdown",
                    options=[{"label": c, "value": c} for c in configs],
                    value=default,
                    clearable=False,
                    style={"minWidth": "320px", "color": "#222"},
                ),
                width="auto",
            ),
        ],
        align="center",
        className="g-2",
    )


navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand(
                [html.I(className="bi bi-diagram-3 me-2"), "CFD Surrogate Workbench"],
                href="/",
            ),
            dbc.Nav(_nav_links(), navbar=True, className="me-auto"),
            _config_selector(),
        ],
        fluid=True,
    ),
    color="primary",
    dark=True,
    className="mb-4",
)

app.layout = dbc.Container(
    [
        dcc.Store(id="config-store", storage_type="session"),
        navbar,
        dbc.Container(dash.page_container, fluid=True),
    ],
    fluid=True,
    className="px-0",
)


@app.callback(Output("config-store", "data"), Input("config-dropdown", "value"))
def _sync_config(value):
    """Mirror the navbar dropdown into the shared session store."""
    return value or runner.DEFAULT_CONFIG


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
