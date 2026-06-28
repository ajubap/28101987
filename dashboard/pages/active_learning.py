"""Active learning page: rank candidate cases by predictive uncertainty."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dash_table, dcc, html

from dashboard import runner
from dashboard.theme import UPLOAD_STYLE, card, command_output, output_panel, page_header, run_button

dash.register_page(__name__, path="/active-learning", name="Active Learning", order=6)


layout = html.Div(
    [
        page_header("Active Learning", "Find the candidate cases whose CFD runs would teach the model the most."),
        card(
            "Rank candidates",
            [
                dcc.Upload(
                    id="cand-upload",
                    children=html.Div(["Drag & drop or ", html.A("select candidate_cases.csv")], className="text-muted"),
                    multiple=False,
                    style=UPLOAD_STYLE,
                ),
                dbc.Row(
                    [
                        dbc.Col([dbc.Label("Candidate CSV path", size="sm"), dbc.Input(id="cand-path", value="data/raw/candidate_cases.csv", size="sm")], md=6),
                        dbc.Col([dbc.Label("Top K", size="sm"), dbc.Input(id="cand-topk", type="number", value=20, min=1, size="sm")], md=3),
                    ],
                    className="g-2 mt-1",
                ),
                dbc.Label("MC dropout samples", size="sm", className="mt-2"),
                dcc.Slider(id="cand-samples", min=5, max=100, step=5, value=20, marks={5: "5", 50: "50", 100: "100"}),
                run_button("cand-run", "Rank candidates"),
                output_panel("cand-output"),
            ],
            "bi-bar-chart-steps",
        ),
        card("Ranking", output_panel("cand-table"), "bi-table"),
    ]
)


@callback(
    Output("cand-output", "children"),
    Output("cand-table", "children"),
    Input("cand-run", "n_clicks"),
    State("cand-upload", "contents"),
    State("cand-upload", "filename"),
    State("cand-path", "value"),
    State("cand-topk", "value"),
    State("cand-samples", "value"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def _rank(_clicks, contents, filename, cand_path, top_k, samples, config_path):
    config_path = config_path or runner.DEFAULT_CONFIG
    cfg = runner.load_cfg(config_path)
    paths = runner.cfg_paths(cfg)

    if contents:
        saved = runner.save_upload(contents, "candidate_cases.csv", paths["raw_dir"])
        cand_path = str(saved.relative_to(runner.PROJECT_ROOT))

    result = runner.run_script(
        [
            "scripts/rank_active_learning.py", "--config", config_path,
            "--candidate_csv", cand_path,
            "--samples", str(int(samples or 20)),
            "--top_k", str(int(top_k or 20)),
        ]
    )

    rank_path = paths["output_dir"] / "active_learning_rank.csv"
    table = dbc.Alert("No ranking produced yet.", color="info")
    if rank_path.exists():
        try:
            import pandas as pd

            df = pd.read_csv(rank_path)
            table = dash_table.DataTable(
                data=df.to_dict("records"),
                columns=[{"name": c, "id": c} for c in df.columns],
                page_size=20,
                sort_action="native",
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "fontSize": "13px"},
            )
        except Exception as exc:
            table = dbc.Alert(f"Could not read ranking: {exc}", color="danger")
    return command_output(result), table
