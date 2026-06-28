from __future__ import annotations

from pathlib import Path
import re
import shutil
import zipfile
from dataclasses import dataclass
from typing import Iterable

from .case_schema import append_case_to_metadata as append_case_to_metadata_for_config

try:
    import pyvista as pv
except Exception:
    pv = None


@dataclass
class ConvertedCase:
    case_id: str
    source_case_file: str
    output_files: list[str]
    combined_file: str | None
    point_arrays: list[str]
    cell_arrays: list[str]
    n_points: int
    n_cells: int


def safe_extract_zip(zip_path: str | Path, extract_dir: str | Path) -> Path:
    """Safely extract a ZIP archive while preventing path traversal."""
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            target = extract_dir / member.filename
            resolved_target = target.resolve()
            resolved_extract = extract_dir.resolve()
            if not str(resolved_target).startswith(str(resolved_extract)):
                raise ValueError(f"Unsafe ZIP member path: {member.filename}")
        zf.extractall(extract_dir)

    return extract_dir


def find_case_files(path: str | Path) -> list[Path]:
    """Find EnSight .case files from a file or directory."""
    path = Path(path)
    if path.is_file() and path.suffix.lower() == ".case":
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.case"))
    return []


def prepare_ensight_input(input_path: str | Path, work_dir: str | Path) -> list[Path]:
    """
    Accept either:
    - a .case file
    - a ZIP file containing the full EnSight folder
    - a directory containing a .case file and companion geometry/variable files
    """
    input_path = Path(input_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if input_path.suffix.lower() == ".zip":
        extract_dir = work_dir / input_path.stem
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        safe_extract_zip(input_path, extract_dir)
        case_files = find_case_files(extract_dir)
    else:
        case_files = find_case_files(input_path)

    if not case_files:
        raise FileNotFoundError(
            "No .case file found. For EnSight, upload the .case file together with "
            "its geometry/variable files, preferably as a ZIP."
        )

    return case_files


def case_file_references(case_file: str | Path) -> list[dict]:
    """
    Best-effort parser for filenames referenced inside an EnSight Gold .case file.

    This is diagnostic only. EnSight syntax can be flexible, so false positives are
    possible, but it catches the common issue: .case file exists while referenced
    geometry/variable files are missing after upload.
    """
    case_file = Path(case_file)
    base = case_file.parent
    refs = []

    if not case_file.exists():
        return refs

    for lineno, raw in enumerate(case_file.read_text(errors="ignore").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue

        left, right = line.split(":", 1)
        left_l = left.lower().strip()
        if left_l in {"format", "type", "description", "number of steps", "filename start number", "filename increment", "time set", "number of files"}:
            continue

        rhs = right.strip()
        if not rhs:
            continue

        # EnSight lines often look like: model: 1 geometry.geo
        # or: scalar per node: 1 temperature temp_****.scl
        tokens = rhs.split()
        if not tokens:
            continue

        candidate = tokens[-1].strip().strip('"').strip("'")
        if not candidate or candidate.replace(".", "", 1).isdigit():
            continue

        # Skip pure labels without file-like content unless they contain wildcard.
        if not any(ch in candidate for ch in [".", "*", "%", "\\", "/"]):
            continue

        candidate_norm = candidate.replace("\\", "/")
        has_wildcard = any(ch in candidate_norm for ch in ["*", "%"])

        if has_wildcard:
            # Convert EnSight-style filename patterns to a broad glob.
            glob_pat = candidate_norm.replace("%", "*")
            matches = sorted(base.glob(glob_pat))
            exists = len(matches) > 0
            resolved = str(base / candidate_norm)
        else:
            p = base / candidate_norm
            exists = p.exists()
            resolved = str(p)
            matches = [p] if exists else []

        refs.append({
            "line": lineno,
            "keyword": left.strip(),
            "reference": candidate,
            "resolved": resolved,
            "exists": exists,
            "matches": [str(m) for m in matches[:5]],
        })

    return refs


def _clean_array_names(dataset):
    """Replace spaces in array names with underscores to make downstream use easier."""
    for store_name in ["point_data", "cell_data", "field_data"]:
        if not hasattr(dataset, store_name):
            continue
        store = getattr(dataset, store_name)
        for name in list(store.keys()):
            new_name = str(name).strip().replace(" ", "_")
            if new_name and new_name != name:
                store[new_name] = store.pop(name)
    return dataset


def _to_unstructured_grid(dataset):
    """Convert PyVista dataset to UnstructuredGrid where possible."""
    if dataset is None:
        return None

    if getattr(dataset, "n_points", 0) == 0 or getattr(dataset, "n_cells", 0) == 0:
        return None

    try:
        dataset = _clean_array_names(dataset)
    except Exception:
        pass

    try:
        if dataset.n_cells > 0 and hasattr(dataset, "cell_data") and len(dataset.cell_data.keys()) > 0:
            try:
                dataset = dataset.cell_data_to_point_data(pass_cell_data=True)
            except TypeError:
                dataset = dataset.cell_data_to_point_data()
    except Exception:
        pass

    if pv is None:
        raise ImportError("pyvista is required for EnSight conversion.")

    if isinstance(dataset, pv.UnstructuredGrid):
        return dataset

    try:
        return dataset.cast_to_unstructured_grid()
    except Exception:
        try:
            return pv.UnstructuredGrid(dataset)
        except Exception:
            return dataset


def _flatten_multiblock(dataset) -> list:
    """Flatten nested PyVista MultiBlock/composite objects into a list of datasets."""
    if pv is None:
        raise ImportError("pyvista is required for EnSight conversion.")

    if dataset is None:
        return []

    if isinstance(dataset, pv.MultiBlock):
        blocks = []
        for i in range(len(dataset)):
            try:
                block = dataset[i]
            except Exception:
                continue
            blocks.extend(_flatten_multiblock(block))
        return blocks

    # Some VTK composite datasets may be wrapped differently depending on VTK/PyVista version.
    if hasattr(dataset, "n_blocks") and not hasattr(dataset, "n_points"):
        blocks = []
        for i in range(int(dataset.n_blocks)):
            try:
                blocks.extend(_flatten_multiblock(dataset[i]))
            except Exception:
                pass
        if blocks:
            return blocks

    return [dataset]


def _reader_enable_all_arrays(reader):
    """Enable arrays if the PyVista reader exposes array-selection methods."""
    for method_name in [
        "enable_all_point_arrays",
        "enable_all_cell_arrays",
        "enable_all_patch_arrays",
        "enable_all_block_arrays",
        "enable_all_part_arrays",
    ]:
        method = getattr(reader, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass


def read_ensight_case(case_file: str | Path, time_index: int | None = None):
    """
    Read an EnSight .case file using PyVista/VTK.

    Uses PyVista's EnSightReader first and falls back to pv.read / vtkGenericEnSightReader.
    """
    if pv is None:
        raise ImportError("pyvista is required. Install with: pip install pyvista vtk")

    case_file = Path(case_file).resolve()

    # Reader route gives us access to time index and array selection.
    reader_errors = []
    try:
        reader = pv.get_reader(str(case_file))
        _reader_enable_all_arrays(reader)

        if time_index is not None:
            for attr_name in ["set_active_time_point", "set_active_time_value"]:
                method = getattr(reader, attr_name, None)
                if callable(method):
                    try:
                        method(int(time_index))
                        break
                    except Exception:
                        pass
            try:
                reader.active_time_point = int(time_index)
            except Exception:
                pass

        data = reader.read()
        if data is not None:
            return data
    except Exception as e:
        reader_errors.append(f"pv.get_reader route failed: {e}")

    try:
        return pv.read(str(case_file))
    except Exception as e:
        reader_errors.append(f"pv.read route failed: {e}")

    try:
        from vtkmodules.vtkIOEnSight import vtkGenericEnSightReader
        vtk_reader = vtkGenericEnSightReader()
        vtk_reader.SetCaseFileName(str(case_file))
        vtk_reader.Update()
        return pv.wrap(vtk_reader.GetOutput())
    except Exception as e:
        reader_errors.append(f"vtkGenericEnSightReader route failed: {e}")

    raise RuntimeError("Could not read EnSight case. " + " | ".join(reader_errors))


def dataset_tree_summary(dataset, max_depth: int = 5, _depth: int = 0, _path: str = "root") -> list[dict]:
    """Return a recursive summary of what PyVista/VTK returned."""
    if dataset is None:
        return [{"path": _path, "type": "None", "n_points": 0, "n_cells": 0, "point_arrays": [], "cell_arrays": []}]

    row = {
        "path": _path,
        "type": type(dataset).__name__,
        "n_points": int(getattr(dataset, "n_points", 0) or 0),
        "n_cells": int(getattr(dataset, "n_cells", 0) or 0),
        "point_arrays": list(getattr(getattr(dataset, "point_data", {}), "keys", lambda: [])()),
        "cell_arrays": list(getattr(getattr(dataset, "cell_data", {}), "keys", lambda: [])()),
    }
    rows = [row]

    if _depth >= max_depth:
        return rows

    if pv is not None and isinstance(dataset, pv.MultiBlock):
        for i in range(len(dataset)):
            try:
                name = dataset.get_block_name(i) or str(i)
            except Exception:
                name = str(i)
            try:
                rows.extend(dataset_tree_summary(dataset[i], max_depth, _depth + 1, f"{_path}/{name}"))
            except Exception as e:
                rows.append({"path": f"{_path}/{name}", "type": f"ERROR: {e}", "n_points": 0, "n_cells": 0, "point_arrays": [], "cell_arrays": []})

    return rows


def convert_ensight_case_to_vtu(
    case_file: str | Path,
    output_dir: str | Path,
    case_id: str | None = None,
    combine_blocks: bool = True,
    time_index: int | None = None,
) -> ConvertedCase:
    """
    Convert an EnSight .case dataset to one or more .vtu files.

    For neural-network use, combine_blocks=True is recommended because the
    MeshGNN workflow expects one mesh file per training case.
    """
    if pv is None:
        raise ImportError("pyvista is required. Install with: pip install pyvista vtk")

    case_file = Path(case_file).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_id = case_id or case_file.stem

    data = read_ensight_case(case_file, time_index=time_index)
    raw_blocks = _flatten_multiblock(data)
    blocks = [_to_unstructured_grid(b) for b in raw_blocks]
    blocks = [b for b in blocks if b is not None and getattr(b, "n_points", 0) > 0 and getattr(b, "n_cells", 0) > 0]

    if not blocks:
        refs = case_file_references(case_file)
        missing = [r for r in refs if not r["exists"]]
        tree = dataset_tree_summary(data)
        raise ValueError(
            "The EnSight file was read, but no valid mesh blocks were found.\n"
            "Most common causes:\n"
            "1) The .case file was uploaded without its referenced geometry/result files.\n"
            "2) The .case file contains absolute/old relative paths that do not exist in the uploaded folder.\n"
            "3) The selected timestep has no geometry. Try --time_index 0.\n"
            "4) The file is a special EnSight variant that VTK/PyVista cannot load fully. Try ParaView export to VTU.\n\n"
            f"Case file: {case_file}\n"
            f"Missing referenced files detected: {missing[:10]}\n"
            f"Reader returned tree summary: {tree[:20]}"
        )

    output_files = []
    combined_file = None

    if combine_blocks:
        mb = pv.MultiBlock(blocks)
        try:
            combined = mb.combine(merge_points=True)
        except TypeError:
            combined = mb.combine()

        combined = _to_unstructured_grid(combined)
        out = output_dir / f"{case_id}.vtu"
        combined.save(out)
        output_files.append(str(out))
        combined_file = str(out)
        summary_mesh = combined
    else:
        summary_mesh = blocks[0]
        for i, block in enumerate(blocks):
            out = output_dir / f"{case_id}_part_{i:03d}.vtu"
            block.save(out)
            output_files.append(str(out))

    return ConvertedCase(
        case_id=case_id,
        source_case_file=str(case_file),
        output_files=output_files,
        combined_file=combined_file,
        point_arrays=list(summary_mesh.point_data.keys()),
        cell_arrays=list(summary_mesh.cell_data.keys()),
        n_points=int(summary_mesh.n_points),
        n_cells=int(summary_mesh.n_cells),
    )


def guess_array_mapping(point_arrays: Iterable[str]) -> dict:
    """
    Best-effort map of solver variable names to standard names used by the package.

    User should verify this mapping before training.
    """
    names = list(point_arrays)
    lower = {name.lower(): name for name in names}

    def find_any(candidates):
        for cand in candidates:
            for lname, original in lower.items():
                if cand in lname:
                    return original
        return None

    return {
        "temperature": find_any(["temperature", "temp", "tfluid", "solid_temp", "wall_temp", "t"]),
        "pressure": find_any(["pressure", "static_pressure", "p_static", "press", "p"]),
        "velocity": find_any(["velocity", "vel", "speed", "u", "v", "w"]),
    }


def append_case_to_metadata(
    metadata_csv: str | Path,
    case_id: str,
    mesh_file: str,
    flow_rate_lpm: float,
    inlet_temp_c: float,
    power_w: float,
    plate_thickness_mm: float,
    target_max_temp_c: float | None = None,
    target_pressure_drop_pa: float | None = None,
):
    """Append or update a row in the dataset metadata CSV."""
    import pandas as pd

    metadata_csv = Path(metadata_csv)
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "case_id": case_id,
        "mesh_file": mesh_file,
        "flow_rate_lpm": flow_rate_lpm,
        "inlet_temp_c": inlet_temp_c,
        "power_w": power_w,
        "plate_thickness_mm": plate_thickness_mm,
        "target_max_temp_c": target_max_temp_c if target_max_temp_c is not None else "",
        "target_pressure_drop_pa": target_pressure_drop_pa if target_pressure_drop_pa is not None else "",
    }

    if metadata_csv.exists():
        df = pd.read_csv(metadata_csv)
        df = df[df["case_id"].astype(str) != str(case_id)]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(metadata_csv, index=False)
    return metadata_csv


def append_case_to_metadata_from_config(
    metadata_csv: str | Path,
    cfg: dict,
    case_id: str,
    mesh_file: str,
    values: dict | None = None,
):
    """Append or update a row using metadata columns declared in config."""
    return append_case_to_metadata_for_config(
        metadata_csv=metadata_csv,
        cfg=cfg,
        case_id=case_id,
        mesh_file=mesh_file,
        values=values or {},
    )
