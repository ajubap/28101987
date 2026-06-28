"""Train page: preprocess meshes into graphs, train, and watch the loss curve."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html

from dashboard import runner
from dashboard.theme import card, command_output, output_panel, page_header, run_button

dash.register_page(__name__, path="/train", name="Train", order=3)


layout = html.Div(
    [
        page_header("Train", "Preprocess validated cases into graphs, then train the surrogate."),
        dbc.Row(
            [
                dbc.Col(
                    card(
                        "1. Preprocess",
                        [
                            html.P("Convert registered meshes into graph .pt files and build splits.", className="text-muted"),
                            run_button("pre-run", "Preprocess dataset"),
                            output_panel("pre-output"),
                        ],
                        "bi-diagram-2",
                    ),
                    md=6,
                ),
                dbc.Col(
                    card(
                        "2. Train",
                        [
                            html.P("Train using the model/loss/optimizer settings in the active config.", className="text-muted"),
                            run_button("train-run", "Train model"),
                            output_panel("train-output"),
                        ],
                        "bi-cpu",
                    ),
                    md=6,
                ),
            ]
        ),
        card(
            "Training curve",
            [
                run_button("curve-refresh", "Refresh curve", color="secondary"),
                dcc.Graph(id="loss-curve"),
            ],
            "bi-graph-up",
        ),
    ]
)


@callback(
    Output("pre-output", "children"),
    Input("pre-run", "n_clicks"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def _preprocess(_clicks, config_path):
    config_path = config_path or runner.DEFAULT_CONFIG
    return command_output(runner.run_script(["scripts/preprocess_cases.py", "--config", config_path]))


@callback(
    Output("train-output", "children"),
    Input("train-run", "n_clicks"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def _train(_clicks, config_path):
    config_path = config_path or runner.DEFAULT_CONFIG
    return command_output(runner.run_script(["scripts/train_meshgnn.py", "--config", config_path]))


@callback(
    Output("loss-curve", "figure"),
    Input("curve-refresh", "n_clicks"),
    Input("config-store", "data"),
)
def _curve(_clicks, config_path):
    config_path = config_path or runner.DEFAULT_CONFIG
    fig = go.Figure()
    fig.update_layout(template="plotly_white", margin=dict(l=40, r=20, t=30, b=40), xaxis_title="epoch", yaxis_title="loss")
    try:
        import pandas as pd

        log_path = runner.cfg_paths(runner.load_cfg(config_path))["output_dir"] / "training_log.csv"
        if not log_path.exists():
            fig.add_annotation(text="No training_log.csv yet — train a model first.", showarrow=False)
            return fig
        log = pd.read_csv(log_path)
        fig.add_trace(go.Scatter(x=log["epoch"], y=log["train_loss"], name="train", mode="lines"))
        fig.add_trace(go.Scatter(x=log["epoch"], y=log["val_loss"], name="val", mode="lines"))
    except Exception as exc:  # pragma: no cover - defensive
        fig.add_annotation(text=f"Could not read training log: {exc}", showarrow=False)
    return fig
