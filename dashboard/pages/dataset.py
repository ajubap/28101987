"""Dataset page: validate the manifest and browse registered cases."""

from __future__ import annotations

import json

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dash_table, html

from dashboard import runner
from dashboard.theme import card, output_panel, page_header, run_button

dash.register_page(__name__, path="/dataset", name="Dataset", order=2)


layout = html.Div(
    [
        page_header("Dataset", "Check that registered cases satisfy the config contract before training."),
        card(
            "Validation",
            [
                dbc.Checklist(
                    options=[{"label": "Deep check: read each mesh and verify point arrays", "value": "deep"}],
                    value=[],
                    id="val-deep",
                    switch=True,
                ),
                run_button("val-run", "Validate dataset"),
                output_panel("val-output"),
            ],
            "bi-clipboard-check",
        ),
        card("Registered cases (cases.csv)", output_panel("cases-table"), "bi-table"),
    ]
)


def _summary_badges(summary):
    return dbc.Row(
        [
            dbc.Col(dbc.Card(dbc.CardBody([html.H4(summary["errors"]), "Errors"]), color="danger" if summary["errors"] else "light", inverse=bool(summary["errors"]))),
            dbc.Col(dbc.Card(dbc.CardBody([html.H4(summary["warnings"]), "Warnings"]), color="warning" if summary["warnings"] else "light", inverse=bool(summary["warnings"]))),
            dbc.Col(dbc.Card(dbc.CardBody([html.H4(summary["total"]), "Checks"]))),
        ],
        className="g-2 mb-3",
    )


@callback(
    Output("val-output", "children"),
    Input("val-run", "n_clicks"),
    State("val-deep", "value"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def _validate(_clicks, deep, config_path):
    config_path = config_path or runner.DEFAULT_CONFIG
    args = ["scripts/validate_dataset.py", "--config", config_path, "--json"]
    if deep and "deep" in deep:
        args.append("--check_mesh_arrays")
    result = runner.run_script(args)

    try:
        payload = json.loads(result["stdout"])
    except (json.JSONDecodeError, KeyError):
        return dbc.Alert(["Validation could not run:", html.Pre(result.get("stderr") or result.get("stdout") or "")], color="danger")

    summary = payload["summary"]
    issues = payload["issues"]
    blocks = [_summary_badges(summary)]
    if summary["errors"]:
        blocks.append(dbc.Alert("Dataset is not ready for preprocessing or training.", color="danger"))
    elif summary["warnings"]:
        blocks.append(dbc.Alert("Dataset can proceed, but review the warnings.", color="warning"))
    else:
        blocks.append(dbc.Alert("Dataset passed the configured readiness checks.", color="success"))

    if issues:
        blocks.append(
            dash_table.DataTable(
                data=issues,
                columns=[{"name": c, "id": c} for c in ("level", "where", "message")],
                style_cell={"textAlign": "left", "fontSize": "13px", "whiteSpace": "normal", "height": "auto"},
                style_data_conditional=[
                    {"if": {"filter_query": '{level} = "error"'}, "backgroundColor": "#f8d7da"},
                    {"if": {"filter_query": '{level} = "warning"'}, "backgroundColor": "#fff3cd"},
                ],
                page_size=15,
            )
        )
    return html.Div(blocks)


@callback(Output("cases-table", "children"), Input("config-store", "data"))
def _cases(config_path):
    config_path = config_path or runner.DEFAULT_CONFIG
    cfg = runner.load_cfg(config_path)
    metadata_csv = runner.cfg_paths(cfg)["metadata_csv"]
    if not metadata_csv.exists():
        return dbc.Alert(f"No cases.csv yet at {metadata_csv}. Add data on the Data page.", color="info")
    try:
        import pandas as pd

        df = pd.read_csv(metadata_csv)
    except Exception as exc:
        return dbc.Alert(f"Could not read {metadata_csv}: {exc}", color="danger")
    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        page_size=12,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "fontSize": "13px"},
    )
