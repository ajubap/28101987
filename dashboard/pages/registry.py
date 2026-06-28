"""Registry page: version a trained model, run a quality gate, browse outputs."""

from __future__ import annotations

import json

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, dash_table, html

from dashboard import runner
from dashboard.theme import card, command_output, output_panel, page_header, run_button

dash.register_page(__name__, path="/registry", name="Registry", order=7)


layout = html.Div(
    [
        page_header("Registry & Outputs", "Sign off, version the model, and inspect produced artifacts."),
        card("Quality gate", output_panel("gate-view"), "bi-shield-check"),
        card(
            "Register model version",
            [
                dbc.Label("Notes", size="sm"),
                dbc.Textarea(id="reg-notes", value="Dashboard-registered model.", style={"height": "80px"}),
                run_button("reg-run", "Register model version"),
                output_panel("reg-output"),
            ],
            "bi-box-seam",
        ),
        card(
            "Output files",
            [
                run_button("files-refresh", "Refresh", color="secondary"),
                html.Div(id="files-table", className="mt-2"),
                dbc.Row(
                    [
                        dbc.Col([dbc.Label("File to download", size="sm"), dbc.Input(id="dl-name", value="validation_report.md", size="sm")], md=6),
                        dbc.Col(run_button("dl-run", "Download", color="secondary"), md=3, className="pt-4"),
                    ],
                    className="g-2 mt-1",
                ),
                dcc.Download(id="dl-file"),
            ],
            "bi-folder2-open",
        ),
    ]
)


@callback(Output("gate-view", "children"), Input("config-store", "data"), Input("files-refresh", "n_clicks"))
def _gate(config_path, _clicks):
    config_path = config_path or runner.DEFAULT_CONFIG
    cfg = runner.load_cfg(config_path)
    paths = runner.cfg_paths(cfg)
    out_dir = paths["output_dir"]
    processed = paths["processed_dir"]

    val = runner.run_script(["scripts/validate_dataset.py", "--config", config_path, "--json"])
    try:
        errors = json.loads(val["stdout"])["summary"]["errors"]
    except Exception:
        errors = None

    def status(ok, blocked_label="BLOCKED"):
        return "PASS" if ok else blocked_label

    checks = [
        {"owner": "Config", "check": "Config declares field targets.", "status": status(bool(cfg.get("data", {}).get("target_field_arrays")), "FAIL")},
        {"owner": "Dataset", "check": "Dataset metadata valid (no errors).", "status": ("PASS" if errors == 0 else "FAIL" if errors else "UNKNOWN")},
        {"owner": "Preprocess", "check": "Processed graph manifest exists.", "status": status((processed / "manifest.csv").exists())},
        {"owner": "Training", "check": "Checkpoint + training log exist.", "status": status((out_dir / "best_meshgnn.pt").exists() and (out_dir / "training_log.csv").exists())},
        {"owner": "Evaluation", "check": "At least one eval_*.json exists.", "status": status(any(out_dir.glob("eval_*.json")))},
        {"owner": "Quality", "check": "Validation report exists.", "status": status((out_dir / "validation_report.md").exists())},
    ]
    table = dash_table.DataTable(
        data=checks,
        columns=[{"name": c, "id": c} for c in ("owner", "check", "status")],
        style_cell={"textAlign": "left", "fontSize": "13px", "whiteSpace": "normal", "height": "auto"},
        style_data_conditional=[
            {"if": {"filter_query": '{status} = "PASS"'}, "backgroundColor": "#d1e7dd"},
            {"if": {"filter_query": '{status} = "FAIL"'}, "backgroundColor": "#f8d7da"},
            {"if": {"filter_query": '{status} = "BLOCKED"'}, "backgroundColor": "#fff3cd"},
        ],
    )
    return table


@callback(
    Output("reg-output", "children"),
    Input("reg-run", "n_clicks"),
    State("reg-notes", "value"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def _register(_clicks, notes, config_path):
    config_path = config_path or runner.DEFAULT_CONFIG
    return command_output(runner.run_script(["scripts/register_model.py", "--config", config_path, "--notes", notes or ""]))


@callback(Output("files-table", "children"), Input("files-refresh", "n_clicks"), Input("config-store", "data"))
def _files(_clicks, config_path):
    config_path = config_path or runner.DEFAULT_CONFIG
    out_dir = runner.cfg_paths(runner.load_cfg(config_path))["output_dir"]
    if not out_dir.exists():
        return dbc.Alert("No outputs yet.", color="info")
    rows = [
        {"path": str(p.relative_to(out_dir)), "size_kb": round(p.stat().st_size / 1024, 2)}
        for p in sorted(out_dir.rglob("*"))
        if p.is_file()
    ]
    if not rows:
        return dbc.Alert("No outputs yet.", color="info")
    return dash_table.DataTable(
        data=rows,
        columns=[{"name": "path", "id": "path"}, {"name": "size_kb", "id": "size_kb"}],
        page_size=15,
        sort_action="native",
        style_cell={"textAlign": "left", "fontSize": "13px"},
    )


@callback(
    Output("dl-file", "data"),
    Input("dl-run", "n_clicks"),
    State("dl-name", "value"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def _download(_clicks, name, config_path):
    config_path = config_path or runner.DEFAULT_CONFIG
    out_dir = runner.cfg_paths(runner.load_cfg(config_path))["output_dir"]
    target = (out_dir / (name or "")).resolve()
    # Guard against path traversal outside the output directory.
    if not str(target).startswith(str(out_dir.resolve())) or not target.is_file():
        return dash.no_update
    return dcc.send_file(str(target))
