from pathlib import Path
import numpy as np
import torch

try:
    import meshio
except Exception:
    meshio = None

try:
    from torch_geometric.data import Data
except Exception:
    Data = None


def _add_undirected_edge(edges, a, b):
    if int(a) != int(b):
        edges.add((int(a), int(b)))
        edges.add((int(b), int(a)))


def edges_from_cells(cells):
    edges = set()

    for block in cells:
        arr = np.asarray(block.data)
        if arr.ndim != 2:
            continue

        nnode = arr.shape[1]

        if nnode == 2:
            for e in arr:
                _add_undirected_edge(edges, e[0], e[1])

        elif nnode in (3, 4):
            for elem in arr:
                for i in range(nnode):
                    for j in range(i + 1, nnode):
                        _add_undirected_edge(edges, elem[i], elem[j])

        elif nnode == 8:
            pairs = [
                (0,1),(1,2),(2,3),(3,0),
                (4,5),(5,6),(6,7),(7,4),
                (0,4),(1,5),(2,6),(3,7),
                (0,2),(4,6),(0,5),(1,4),(2,7),(3,6)
            ]
            for elem in arr:
                for i, j in pairs:
                    _add_undirected_edge(edges, elem[i], elem[j])
        else:
            for elem in arr:
                for i in range(nnode):
                    for j in range(i + 1, nnode):
                        _add_undirected_edge(edges, elem[i], elem[j])

    if not edges:
        raise ValueError("No graph edges could be constructed from mesh cells.")

    return np.array(sorted(edges), dtype=np.int64).T


def point_array(mesh, name, n, required=False, role="array"):
    if name in mesh.point_data:
        a = np.asarray(mesh.point_data[name])
        if a.ndim == 1:
            return a.reshape(n, 1).astype(np.float32)
        return a.astype(np.float32)
    if required:
        available = sorted(mesh.point_data.keys())
        raise ValueError(f"Missing required {role} point array '{name}'. Available point arrays: {available}")
    return np.zeros((n, 1), dtype=np.float32)


def component_names(base_name, array):
    width = 1 if array.ndim == 1 else int(array.shape[1])
    if width == 1:
        return [base_name]
    suffixes = ["x", "y", "z"] if width == 3 else [str(i) for i in range(width)]
    return [f"{base_name}.{suffix}" for suffix in suffixes]


def mesh_file_to_graph(
    mesh_path,
    global_features,
    point_feature_arrays,
    target_field_arrays,
    scalar_targets=None,
    case_id=None,
    require_target_fields=True,
):
    if meshio is None:
        raise ImportError("meshio is required. Install with: pip install meshio")
    if Data is None:
        raise ImportError("torch_geometric is required. Install PyTorch Geometric.")

    mesh_path = Path(mesh_path)
    mesh = meshio.read(mesh_path)

    points = np.asarray(mesh.points, dtype=np.float32)
    n = points.shape[0]
    if points.shape[1] == 2:
        points = np.pad(points, ((0, 0), (0, 1)), mode="constant")

    edge_index = edges_from_cells(mesh.cells)
    src, dst = edge_index[0], edge_index[1]

    rel = points[dst] - points[src]
    dist = np.linalg.norm(rel, axis=1, keepdims=True)
    edge_attr = np.concatenate([rel, dist], axis=1).astype(np.float32)

    global_vec = np.array([float(global_features[k]) for k in global_features.keys()], dtype=np.float32)
    global_repeated = np.repeat(global_vec.reshape(1, -1), n, axis=0)

    point_features = [point_array(mesh, name, n, required=True, role="feature") for name in point_feature_arrays]
    point_feature_names = []
    for name, array in zip(point_feature_arrays, point_features):
        point_feature_names.extend(component_names(name, array))
    x = np.concatenate([points, global_repeated] + point_features, axis=1).astype(np.float32)

    target_fields = [
        point_array(mesh, name, n, required=require_target_fields, role="target")
        for name in target_field_arrays
    ]
    target_field_names = []
    for name, array in zip(target_field_arrays, target_fields):
        target_field_names.extend(component_names(name, array))
    y = np.concatenate(target_fields, axis=1).astype(np.float32) if target_fields else np.zeros((n, 0), dtype=np.float32)

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.float32),
        pos=torch.tensor(points, dtype=torch.float32),
        globals=torch.tensor(global_vec.reshape(1, -1), dtype=torch.float32),
    )

    if scalar_targets:
        scalar_names = list(scalar_targets.keys())
        data.y_scalar = torch.tensor([[float(scalar_targets[k]) for k in scalar_names]], dtype=torch.float32)
        data.target_scalar_names = scalar_names
    else:
        data.target_scalar_names = []

    data.case_id = case_id or mesh_path.stem
    data.mesh_path = str(mesh_path)
    data.global_feature_names = list(global_features.keys())
    data.point_feature_names = point_feature_names
    data.x_feature_names = ["x", "y", "z"] + list(global_features.keys()) + point_feature_names
    data.target_field_names = target_field_names

    return data
