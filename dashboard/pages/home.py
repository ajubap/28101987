"""Home page: the active problem contract and the recommended workflow."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, callback, html

from dashboard import runner
from dashboard.theme import card, page_header

dash.register_page(__name__, path="/", name="Overview", order=0)

WORKFLOW = [
    ("1. Data", "Generate synthetic data, upload VTU/VTK, or convert EnSight exports."),
    ("2. Dataset", "Validate the manifest and the per-mesh arrays against the config."),
    ("3. Train", "Preprocess meshes into graphs, then train the surrogate."),
    ("4. Evaluate", "Score the held-out split and produce a validation report."),
    ("5. Inference", "Predict fields/KPIs on new meshes, with optional uncertainty."),
    ("6. Active Learning", "Rank candidate cases by predictive uncertainty."),
    ("7. Registry", "Version the trained model and browse output artifacts."),
]


def _contract_table(cfg):
    data = cfg.get("data", {})
    rows = [
        ("Global inputs", data.get("global_feature_columns", [])),
        ("Point features", data.get("point_feature_arrays", [])),
        ("Field targets", data.get("target_field_arrays", [])),
        ("Scalar targets", data.get("target_scalar_columns", [])),
    ]
    body = [
        html.Tr([html.Th(name, style={"width": "30%"}), html.Td(", ".join(values) or "—")])
        for name, values in rows
    ]
    return dbc.Table([html.Tbody(body)], bordered=True, hover=True, size="sm")


def _model_summary(cfg):
    model = cfg.get("model", {})
    items = [html.Tr([html.Th(k, style={"width": "30%"}), html.Td(str(v))]) for k, v in model.items()]
    return dbc.Table([html.Tbody(items)], bordered=True, hover=True, size="sm")


layout = html.Div(
    [
        page_header(
            "General CFD Surrogate Workbench",
            "Train MeshGNN surrogates for any CFD problem. Everything below is driven "
            "by the config selected in the top-right dropdown.",
        ),
        dbc.Row(
            [
                dbc.Col(card("Problem contract", html.Div(id="home-contract"), "bi-file-earmark-text"), md=6),
                dbc.Col(card("Model", html.Div(id="home-model"), "bi-cpu"), md=6),
            ]
        ),
        card(
            "Workflow",
            dbc.ListGroup(
                [
                    dbc.ListGroupItem([html.Strong(step + ". "), desc])
                    for step, desc in WORKFLOW
                ],
                flush=True,
            ),
            "bi-list-ol",
        ),
    ]
)


@callback(
    Output("home-contract", "children"),
    Output("home-model", "children"),
    Input("config-store", "data"),
)
def _render(config_path):
    config_path = config_path or runner.DEFAULT_CONFIG
    try:
        cfg = runner.load_cfg(config_path)
    except Exception as exc:
        msg = dbc.Alert(f"Could not load config {config_path}: {exc}", color="danger")
        return msg, ""
    return _contract_table(cfg), _model_summary(cfg)
