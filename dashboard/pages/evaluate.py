"""Evaluate page: score a split and render the validation report."""

from __future__ import annotations

import json

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dash_table, dcc, html

from dashboard import runner
from dashboard.theme import card, command_output, output_panel, page_header, run_button

dash.register_page(__name__, path="/evaluate", name="Evaluate", order=4)


layout = html.Div(
    [
        page_header("Evaluate", "Score the trained surrogate and build a traceable validation report."),
        card(
            "Run evaluation",
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Split", size="sm"),
                                dcc.Dropdown(
                                    id="eval-split",
                                    options=[{"label": s, "value": s} for s in ("test", "val", "train")],
                                    value="test",
                                    clearable=False,
                                ),
                            ],
                            md=3,
                        )
                    ]
                ),
                html.Div(
                    [
                        run_button("eval-run", "Run evaluation"),
                        run_button("report-run", "Generate validation report", color="secondary"),
                    ],
                    className="d-flex gap-2",
                ),
                output_panel("eval-output"),
            ],
            "bi-clipboard-data",
        ),
        card("Metrics", output_panel("eval-metrics"), "bi-rulers"),
        card("Validation report", output_panel("report-view"), "bi-file-earmark-text"),
    ]
)


@callback(
    Output("eval-output", "children"),
    Output("eval-metrics", "children"),
    Input("eval-run", "n_clicks"),
    State("eval-split", "value"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def _evaluate(_clicks, split, config_path):
    config_path = config_path or runner.DEFAULT_CONFIG
    result = runner.run_script(["scripts/evaluate_meshgnn.py", "--config", config_path, "--split", split])

    rows = []
    out_dir = runner.cfg_paths(runner.load_cfg(config_path))["output_dir"]
    eval_path = out_dir / f"eval_{split}.json"
    if eval_path.exists():
        try:
            payload = json.loads(eval_path.read_text(encoding="utf-8"))
            for batch_idx, item in enumerate(payload):
                for group, metrics_group in item.items():
                    for output_name, metrics in metrics_group.items():
                        rows.append({"batch": batch_idx, "group": group, "output": output_name, **metrics})
        except Exception:
            pass

    metrics_view = (
        dash_table.DataTable(
            data=rows,
            columns=[{"name": c, "id": c} for c in rows[0].keys()],
            page_size=15,
            style_cell={"textAlign": "left", "fontSize": "13px"},
            style_table={"overflowX": "auto"},
        )
        if rows
        else dbc.Alert("No metrics yet — run evaluation after training.", color="info")
    )
    return command_output(result), metrics_view


@callback(
    Output("report-view", "children"),
    Input("report-run", "n_clicks"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def _report(_clicks, config_path):
    config_path = config_path or runner.DEFAULT_CONFIG
    result = runner.run_script(["scripts/generate_validation_report.py", "--config", config_path])
    report_path = runner.cfg_paths(runner.load_cfg(config_path))["output_dir"] / "validation_report.md"
    if report_path.exists():
        return html.Div([command_output(result), dcc.Markdown(report_path.read_text(encoding="utf-8"))])
    return command_output(result)
