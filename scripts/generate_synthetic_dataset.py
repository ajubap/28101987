"""Generate a learnable synthetic dataset for *any* problem config.

Unlike ``generate_synthetic_coldplate.py`` (which is hard-wired to the cold-plate
problem), this generator reads the active YAML config and produces a dataset that
matches whatever the config declares:

* one ``.vtu`` mesh per case, containing every ``point_feature_arrays`` entry
  (as geometric boundary masks) and every ``target_field_arrays`` entry
  (scalar fields are 1 component, vector-like fields are 3 components),
* a ``cases.csv`` with the ``case_id`` / ``mesh_file`` columns plus every
  ``global_feature_columns`` and ``target_scalar_columns`` value.

The synthetic fields are smooth functions of the node coordinates and the
per-case global inputs, so a surrogate has a genuine signal to learn. This is a
demo/smoke-test dataset, not a substitute for real CFD data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import meshio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physicsai_gnn.config import load_config
from physicsai_gnn.case_schema import (
    case_id_column,
    mesh_file_column,
    global_feature_columns,
    point_feature_arrays,
    target_field_arrays,
    target_scalar_columns,
)

# Names that should be treated as 3-component vector fields.
VECTOR_HINTS = ("velocity", "vel", "displacement", "momentum", "flux", "gradient")


def is_vector_field(name: str) -> bool:
    lname = name.lower()
    return any(hint in lname for hint in VECTOR_HINTS)


def make_grid(nx: int, ny: int, length: float, width: float):
    xs = np.linspace(0.0, length, nx)
    ys = np.linspace(0.0, width, ny)
    points = np.array([[x, y, 0.0] for y in ys for x in xs], dtype=np.float32)
    quads = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            quads.append([n0, n0 + 1, n0 + nx + 1, n0 + nx])
    return points, np.array(quads, dtype=np.int64)


def region_mask(name: str, x: np.ndarray, y: np.ndarray, L: float, W: float, index: int) -> np.ndarray:
    """Assign each declared point feature to a plausible geometric region.

    The mapping is driven by common boundary-condition naming, with an
    index-based fallback so any name still produces a usable mask.
    """
    lname = name.lower()
    if "inlet" in lname or lname.startswith("in"):
        mask = x < 0.02 * L
    elif "outlet" in lname or lname.startswith("out"):
        mask = x > 0.98 * L
    elif "wall" in lname or "noslip" in lname:
        mask = (y < 0.02 * W) | (y > 0.98 * W)
    elif "heat" in lname or "source" in lname or "sink" in lname:
        mask = (x > 0.38 * L) & (x < 0.72 * L) & (y > 0.30 * W) & (y < 0.70 * W)
    elif "symmetry" in lname or "sym" in lname:
        mask = np.abs(y - 0.5 * W) < 0.02 * W
    else:
        # Fallback: split the domain into vertical bands by feature index.
        lo = (index % 4) / 4.0
        hi = lo + 0.25
        mask = (x >= lo * L) & (x < hi * L)
    return mask.astype(np.float32)


def global_value(name: str, rng: np.random.Generator) -> float:
    """Draw a plausible value for a declared global input from its name."""
    lname = name.lower()
    if "temp" in lname:
        return float(rng.uniform(15.0, 45.0))
    if "fraction" in lname or "ratio" in lname:
        return float(rng.uniform(0.05, 0.95))
    if "angle" in lname or "deg" in lname:
        return float(rng.uniform(-10.0, 10.0))
    if "reynolds" in lname or lname == "re":
        return float(rng.uniform(1e3, 1e5))
    if "mach" in lname:
        return float(rng.uniform(0.1, 0.8))
    if "power" in lname:
        return float(rng.choice([16.0, 24.0, 32.0, 40.0]))
    return float(rng.uniform(0.3, 2.0))


def scalar_field(x, y, L, W, drive: float) -> np.ndarray:
    """A smooth scalar field with a clear dependence on the global drive."""
    base = drive * (0.25 + np.exp(-(((x - 0.55 * L) ** 2) / (2 * (0.18 * L) ** 2))))
    ripple = 1.5 * np.sin(2 * np.pi * x / L) * np.cos(np.pi * y / W)
    gradient = drive * (1.0 - x / L)
    return (base + ripple + gradient).astype(np.float32)


def vector_field(x, y, L, W, drive: float) -> np.ndarray:
    vx = drive * (1.0 - 0.5 * (y / W)) * (1.0 - 0.3 * np.sin(np.pi * x / L))
    vy = drive * 0.4 * np.sin(2 * np.pi * x / L) * (0.5 - y / W)
    vz = np.zeros_like(x)
    return np.stack([vx, vy, vz], axis=1).astype(np.float32)


def scalar_target(name: str, fields: dict[str, np.ndarray], drive: float) -> float:
    """Derive a scalar KPI from the generated fields, keyed off its name."""
    lname = name.lower()
    first_field = next(iter(fields.values())) if fields else None

    if first_field is not None:
        flat = first_field.reshape(first_field.shape[0], -1)[:, 0]
        if "max" in lname or "peak" in lname or "hotspot" in lname:
            return float(flat.max())
        if "min" in lname:
            return float(flat.min())
        if "drop" in lname or "delta" in lname or "dp" in lname:
            return float(flat.max() - flat.min())
        if "mean" in lname or "avg" in lname:
            return float(flat.mean())

    if "drag" in lname:
        return float(0.2 + 0.05 * drive)
    if "lift" in lname:
        return float(0.5 * drive)
    return float(drive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/generic_cfd_template.yaml")
    parser.add_argument("--num_cases", type=int, default=60)
    parser.add_argument("--out_dir", default=None, help="Override raw_dir from the config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir or cfg["paths"]["raw_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_csv = Path(cfg["paths"]["metadata_csv"])
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)

    globals_cfg = global_feature_columns(cfg)
    points_cfg = point_feature_arrays(cfg)
    fields_cfg = target_field_arrays(cfg)
    scalars_cfg = target_scalar_columns(cfg)

    seed = int(cfg.get("project", {}).get("seed", 42))
    rng = np.random.default_rng(seed)
    rows = []

    for k in range(args.num_cases):
        case_id = f"case_{k + 1:04d}"

        nx = int(rng.integers(34, 54))
        ny = int(rng.integers(14, 24))
        length = float(rng.uniform(90, 115))
        width = float(rng.uniform(24, 36))
        points, quads = make_grid(nx, ny, length, width)

        x, y = points[:, 0], points[:, 1]
        L, W = x.max(), y.max()

        global_values = {name: global_value(name, rng) for name in globals_cfg}
        # A single scalar "drive" couples the globals into the field signal.
        drive = float(np.mean(list(global_values.values()))) if global_values else 1.0

        point_data: dict[str, np.ndarray] = {}
        for i, name in enumerate(points_cfg):
            point_data[name] = region_mask(name, x, y, L, W, i)

        generated_fields: dict[str, np.ndarray] = {}
        for name in fields_cfg:
            if is_vector_field(name):
                arr = vector_field(x, y, L, W, drive)
            else:
                arr = scalar_field(x, y, L, W, drive)
            point_data[name] = arr
            generated_fields[name] = arr

        mesh_file = f"{case_id}.vtu"
        meshio.write(
            out_dir / mesh_file,
            meshio.Mesh(points=points, cells=[("quad", quads)], point_data=point_data),
        )

        row = {case_id_column(cfg): case_id, mesh_file_column(cfg): mesh_file}
        row.update(global_values)
        for name in scalars_cfg:
            row[name] = scalar_target(name, generated_fields, drive)
        rows.append(row)

    pd.DataFrame(rows).to_csv(metadata_csv, index=False)
    print(
        f"Wrote {len(rows)} synthetic cases to {out_dir} "
        f"and metadata to {metadata_csv}\n"
        f"  globals={globals_cfg}\n  point_features={points_cfg}\n"
        f"  field_targets={fields_cfg}\n  scalar_targets={scalars_cfg}"
    )


if __name__ == "__main__":
    main()
