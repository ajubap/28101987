"""Inference page: predict fields/KPIs on a new mesh, with optional uncertainty."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback, dcc, html

from dashboard import runner
from dashboard.theme import card, command_output, output_panel, page_header, run_button

dash.register_page(__name__, path="/inference", name="Inference", order=5)


layout = html.Div(
    [
        page_header("Inference", "Predict on a new mesh using the config-defined CFD inputs."),
        card(
            "Inputs",
            [
                dbc.Row(
                    [
                        dbc.Col([dbc.Label("Mesh path", size="sm"), dbc.Input(id="inf-mesh", value="data/raw/case_0001.vtu", size="sm")], md=6),
                        dbc.Col([dbc.Label("Export VTU path", size="sm"), dbc.Input(id="inf-export", value="outputs/prediction.vtu", size="sm")], md=6),
                    ],
                    className="g-2",
                ),
                html.H6("Global inputs", className="mt-3"),
                html.Div(id="inf-globals"),
                html.H6("Uncertainty", className="mt-3"),
                dbc.Label("MC dropout samples", size="sm"),
                dcc.Slider(id="inf-samples", min=5, max=100, step=5, value=20, marks={5: "5", 50: "50", 100: "100"}),
                html.Div(
                    [
                        run_button("inf-run", "Deterministic inference"),
                        run_button("unc-run", "Uncertainty inference", color="secondary"),
                    ],
                    className="d-flex gap-2 mt-2",
                ),
                output_panel("inf-output"),
            ],
            "bi-magic",
        ),
    ]
)


@callback(Output("inf-globals", "children"), Input("config-store", "data"))
def _render_globals(config_path):
    cfg = runner.load_cfg(config_path or runner.DEFAULT_CONFIG)
    globals_cfg = cfg.get("data", {}).get("global_feature_columns", [])
    if not globals_cfg:
        return html.P("This config declares no global inputs.", className="text-muted")
    return dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Label(name, size="sm"),
                    dbc.Input(id={"type": "infer-global", "field": name}, type="number", value=0.0, size="sm"),
                ],
                md=3,
            )
            for name in globals_cfg
        ],
        className="g-2",
    )


def _global_values(values, ids):
    return {ident["field"]: val for val, ident in zip(values or [], ids or []) if val not in (None, "")}


@callback(
    Output("inf-output", "children"),
    Input("inf-run", "n_clicks"),
    Input("unc-run", "n_clicks"),
    State("inf-mesh", "value"),
    State("inf-export", "value"),
    State("inf-samples", "value"),
    State({"type": "infer-global", "field": ALL}, "value"),
    State({"type": "infer-global", "field": ALL}, "id"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def _infer(_n1, _n2, mesh, export, samples, g_values, g_ids, config_path):
    trigger = dash.ctx.triggered_id
    config_path = config_path or runner.DEFAULT_CONFIG
    if not mesh:
        return dbc.Alert("Provide a mesh path.", color="warning")

    global_args = runner.global_cli_args(_global_values(g_values, g_ids))
    if trigger == "unc-run":
        args = [
            "scripts/uncertainty_infer.py", "--config", config_path,
            "--mesh", mesh, *global_args,
            "--samples", str(int(samples or 20)),
        ]
    else:
        args = ["scripts/infer_case.py", "--config", config_path, "--mesh", mesh, *global_args]
    if export:
        args += ["--export_vtu", export]
    return command_output(runner.run_script(args))
