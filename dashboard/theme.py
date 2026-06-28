"""Shared presentational helpers so every page looks consistent."""

from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
from dash import dcc, html

# Stylesheets used by the app (imported in app.py).
EXTERNAL_STYLESHEETS = [dbc.themes.FLATLY, dbc.icons.BOOTSTRAP]


def page_header(title: str, subtitle: str = "") -> html.Div:
    children: list[Any] = [html.H3(title, className="mb-1")]
    if subtitle:
        children.append(html.P(subtitle, className="text-muted"))
    return html.Div(children, className="mb-3")


def card(title: str, body, icon: str | None = None) -> dbc.Card:
    header_children: list[Any] = []
    if icon:
        header_children.append(html.I(className=f"bi {icon} me-2"))
    header_children.append(title)
    return dbc.Card(
        [
            dbc.CardHeader(header_children),
            dbc.CardBody(body),
        ],
        className="mb-4 shadow-sm",
    )


def run_button(button_id: str, label: str, color: str = "primary") -> dbc.Button:
    return dbc.Button(
        [html.I(className="bi bi-play-fill me-1"), label],
        id=button_id,
        color=color,
        className="mt-2",
    )


def command_output(result: dict | None):
    """Render the dict returned by ``runner.run_script`` as styled output."""
    if not result:
        return ""
    ok = result.get("returncode") == 0
    badge = dbc.Badge(
        "Success" if ok else f"Failed (code {result.get('returncode')})",
        color="success" if ok else "danger",
        className="mb-2",
    )
    blocks: list[Any] = [badge, html.Pre(result.get("cmd", ""), className="small text-muted")]
    if result.get("stdout"):
        blocks.append(html.Pre(result["stdout"], className="bg-light p-2 border rounded small"))
    if result.get("stderr"):
        blocks.append(
            html.Pre(
                result["stderr"],
                className="bg-danger-subtle text-danger p-2 border rounded small",
            )
        )
    return html.Div(blocks)


def output_panel(panel_id: str):
    """A spinner-wrapped container that a callback fills with command output."""
    return dbc.Spinner(html.Div(id=panel_id), color="primary", size="sm")
