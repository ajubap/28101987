from pathlib import Path
import numpy as np

try:
    import meshio
except Exception:
    meshio = None


def _component_names(names, width, prefix):
    names = list(names or [])
    if len(names) != width:
        names = names + [f"{prefix}_{i}" for i in range(len(names), width)]
    return names[:width]


def export_prediction_vtu(
    original_mesh_path,
    output_vtu_path,
    field_mean,
    field_names,
    field_std=None,
    scalar_mean=None,
    scalar_std=None,
    scalar_names=None,
):
    if meshio is None:
        raise ImportError("meshio is required for VTU export.")

    original_mesh_path = Path(original_mesh_path)
    output_vtu_path = Path(output_vtu_path)
    output_vtu_path.parent.mkdir(parents=True, exist_ok=True)

    mesh = meshio.read(original_mesh_path)

    field_mean_np = field_mean.detach().cpu().numpy()
    field_names = _component_names(field_names, field_mean_np.shape[1], "field")
    for i, name in enumerate(field_names):
        mesh.point_data[f"pred_{name}"] = field_mean_np[:, i]

    if field_std is not None:
        field_std_np = field_std.detach().cpu().numpy()
        for i, name in enumerate(field_names):
            mesh.point_data[f"uncertainty_std_{name}"] = field_std_np[:, i]

    # Store scalar summary as field_data for visibility in VTK metadata.
    scalar_names = scalar_names or []
    if scalar_mean is not None and scalar_names:
        sm = np.asarray(scalar_mean.detach().cpu().numpy()).reshape(-1)
        ss = np.asarray(scalar_std.detach().cpu().numpy()).reshape(-1) if scalar_std is not None else None
        scalar_names = _component_names(scalar_names, len(sm), "scalar")

        for i, name in enumerate(scalar_names):
            # meshio field_data expects name -> [value, num_components].
            mesh.field_data[f"pred_{name}"] = np.array([float(sm[i]), 1])
            if ss is not None:
                mesh.field_data[f"uncertainty_std_{name}"] = np.array([float(ss[i]), 1])

    meshio.write(output_vtu_path, mesh)
    return output_vtu_path
