"""Data page: synthetic generation, VTU/VTK upload, and EnSight conversion."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback, dcc, html

from dashboard import runner
from dashboard.theme import UPLOAD_STYLE, card, command_output, output_panel, page_header, run_button

dash.register_page(__name__, path="/data", name="Data", order=1)


def _meta_inputs(cfg, include_scalars=True):
    data = cfg.get("data", {})
    blocks = []
    globals_cfg = data.get("global_feature_columns", [])
    scalars_cfg = data.get("target_scalar_columns", []) if include_scalars else []

    if globals_cfg:
        blocks.append(html.H6("Global inputs", className="mt-2"))
        blocks.append(
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label(name, size="sm"),
                            dbc.Input(
                                id={"type": "meta", "kind": "global", "field": name},
                                type="number",
                                value=0.0,
                                size="sm",
                            ),
                        ],
                        md=3,
                    )
                    for name in globals_cfg
                ],
                className="g-2",
            )
        )
    if scalars_cfg:
        blocks.append(html.H6("Scalar targets (optional)", className="mt-3"))
        blocks.append(
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label(name, size="sm"),
                            dbc.Input(
                                id={"type": "meta", "kind": "scalar", "field": name},
                                type="number",
                                placeholder="leave blank if unknown",
                                size="sm",
                            ),
                        ],
                        md=3,
                    )
                    for name in scalars_cfg
                ],
                className="g-2",
            )
        )
    if not blocks:
        blocks.append(html.P("This config declares no global inputs or scalar targets.", className="text-muted"))
    return blocks


layout = html.Div(
    [
        page_header("Data", "Get meshes into the dataset, then register them with their CFD inputs."),
        card(
            "A. Generate synthetic dataset",
            [
                html.P(
                    "Builds a learnable demo dataset matching the active config "
                    "(any globals / point features / field & scalar targets). "
                    "Great for smoke-testing the full pipeline.",
                    className="text-muted",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Number of cases", size="sm"),
                                dbc.Input(id="synth-num", type="number", value=60, min=10, step=10, size="sm"),
                            ],
                            md=3,
                        )
                    ]
                ),
                run_button("synth-run", "Generate synthetic data"),
                output_panel("synth-output"),
            ],
            "bi-magic",
        ),
        card(
            "Case metadata",
            html.Div(id="meta-inputs"),
            "bi-input-cursor-text",
        ),
        card(
            "B. Upload VTU/VTK and register",
            [
                dcc.Upload(
                    id="vtu-upload",
                    children=html.Div(["Drag & drop or ", html.A("select .vtu/.vtk files")], className="text-muted"),
                    multiple=True,
                    style=UPLOAD_STYLE,
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Case ID prefix", size="sm"),
                                dbc.Input(id="vtu-prefix", value="case", size="sm"),
                            ],
                            md=3,
                        )
                    ],
                    className="mt-2",
                ),
                run_button("vtu-run", "Save uploads and register", color="secondary"),
                output_panel("vtu-output"),
            ],
            "bi-upload",
        ),
        card(
            "C. Convert EnSight and register",
            [
                dcc.Upload(
                    id="ensight-upload",
                    children=html.Div(["Drag & drop or ", html.A("select an EnSight .zip / .case")], className="text-muted"),
                    multiple=False,
                    style=UPLOAD_STYLE,
                ),
                dbc.Row(
                    [
                        dbc.Col([dbc.Label("Case ID", size="sm"), dbc.Input(id="ens-caseid", value="case_001", size="sm")], md=3),
                        dbc.Col([dbc.Label("Time index (optional)", size="sm"), dbc.Input(id="ens-time", type="number", size="sm")], md=3),
                        dbc.Col(
                            dbc.Checklist(
                                options=[{"label": "Combine parts into one VTU", "value": "combine"}],
                                value=["combine"],
                                id="ens-combine",
                                switch=True,
                            ),
                            md=3,
                            className="pt-4",
                        ),
                    ],
                    className="mt-2",
                ),
                run_button("ens-run", "Convert and register", color="secondary"),
                output_panel("ensight-output"),
            ],
            "bi-box-arrow-in-down",
        ),
    ]
)


@callback(Output("meta-inputs", "children"), Input("config-store", "data"))
def _render_meta(config_path):
    cfg = runner.load_cfg(config_path or runner.DEFAULT_CONFIG)
    return _meta_inputs(cfg)


def _collect_meta(values, ids, kind):
    out = {}
    for value, ident in zip(values or [], ids or []):
        if ident.get("kind") != kind:
            continue
        if value in (None, ""):
            continue
        out[ident["field"]] = value
    return out


@callback(
    Output("synth-output", "children"),
    Input("synth-run", "n_clicks"),
    State("synth-num", "value"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def _run_synth(_clicks, num_cases, config_path):
    config_path = config_path or runner.DEFAULT_CONFIG
    result = runner.run_script(
        ["scripts/generate_synthetic_dataset.py", "--config", config_path, "--num_cases", str(int(num_cases or 60))]
    )
    return command_output(result)


@callback(
    Output("vtu-output", "children"),
    Input("vtu-run", "n_clicks"),
    State("vtu-upload", "contents"),
    State("vtu-upload", "filename"),
    State("vtu-prefix", "value"),
    State({"type": "meta", "kind": ALL, "field": ALL}, "value"),
    State({"type": "meta", "kind": ALL, "field": ALL}, "id"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def _run_vtu(_clicks, contents, filenames, prefix, meta_values, meta_ids, config_path):
    if not contents:
        return dbc.Alert("Upload at least one .vtu/.vtk file first.", color="warning")
    config_path = config_path or runner.DEFAULT_CONFIG
    cfg = runner.load_cfg(config_path)
    paths = runner.cfg_paths(cfg)
    raw_dir = paths["raw_dir"]

    globals_meta = _collect_meta(meta_values, meta_ids, "global")
    scalars_meta = _collect_meta(meta_values, meta_ids, "scalar")
    values = {**globals_meta, **scalars_meta}

    try:
        from physicsai_gnn.case_schema import append_case_to_metadata

        rows = []
        many = len(contents) > 1
        for i, (content, fname) in enumerate(zip(contents, filenames), start=1):
            saved = runner.save_upload(content, fname, raw_dir)
            case_id = f"{prefix}_{i:03d}" if many else prefix
            append_case_to_metadata(
                metadata_csv=paths["metadata_csv"],
                cfg=cfg,
                case_id=case_id,
                mesh_file=saved.name,
                values=values,
            )
            rows.append(f"{case_id} -> {saved.name}")
    except Exception as exc:
        return dbc.Alert(f"Registration failed: {exc}", color="danger")

    return dbc.Alert([html.Strong("Registered:"), html.Ul([html.Li(r) for r in rows])], color="success")


@callback(
    Output("ensight-output", "children"),
    Input("ens-run", "n_clicks"),
    State("ensight-upload", "contents"),
    State("ensight-upload", "filename"),
    State("ens-caseid", "value"),
    State("ens-time", "value"),
    State("ens-combine", "value"),
    State({"type": "meta", "kind": ALL, "field": ALL}, "value"),
    State({"type": "meta", "kind": ALL, "field": ALL}, "id"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def _run_ensight(_clicks, content, filename, case_id, time_index, combine, meta_values, meta_ids, config_path):
    if not content:
        return dbc.Alert("Upload an EnSight .zip or .case first.", color="warning")
    config_path = config_path or runner.DEFAULT_CONFIG
    cfg = runner.load_cfg(config_path)
    paths = runner.cfg_paths(cfg)
    upload_dir = paths["raw_dir"] / "_ensight_uploads"
    saved = runner.save_upload(content, filename, upload_dir)

    args = [
        "scripts/convert_ensight.py",
        "--config", config_path,
        "--input", str(saved),
        "--case_id", case_id or "case_001",
        "--register",
    ]
    if not (combine and "combine" in combine):
        args.append("--no_combine")
    if time_index not in (None, ""):
        args += ["--time_index", str(int(time_index))]

    globals_meta = _collect_meta(meta_values, meta_ids, "global")
    scalars_meta = _collect_meta(meta_values, meta_ids, "scalar")
    args += runner.global_cli_args(globals_meta)
    for name, value in scalars_meta.items():
        args += ["--scalar-target", f"{name}={value}"]

    return command_output(runner.run_script(args))
