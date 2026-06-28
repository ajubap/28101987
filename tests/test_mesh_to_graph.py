import importlib.util

import numpy as np
import pytest

from physicsai_gnn.mesh_to_graph import component_names, point_array


def test_component_names_expands_vector_fields():
    array = np.zeros((4, 3), dtype=np.float32)
    assert component_names("velocity", array) == ["velocity.x", "velocity.y", "velocity.z"]


def test_missing_required_point_array_fails_loudly():
    class Mesh:
        point_data = {}

    with pytest.raises(ValueError, match="Missing required target point array"):
        point_array(Mesh(), "pressure", 3, required=True, role="target")


@pytest.mark.skipif(
    importlib.util.find_spec("meshio") is None or importlib.util.find_spec("torch_geometric") is None,
    reason="meshio and torch_geometric are required for graph conversion smoke test.",
)
def test_mesh_file_to_graph_expands_vector_target_names(tmp_path):
    import meshio

    from physicsai_gnn.mesh_to_graph import mesh_file_to_graph

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    cells = [("quad", np.array([[0, 1, 2, 3]], dtype=np.int64))]
    mesh = meshio.Mesh(
        points,
        cells,
        point_data={
            "wall_mask": np.array([1, 0, 0, 1], dtype=np.float32),
            "velocity": np.ones((4, 3), dtype=np.float32),
        },
    )
    mesh_path = tmp_path / "case.vtu"
    meshio.write(mesh_path, mesh)

    graph = mesh_file_to_graph(
        mesh_path,
        global_features={"reynolds": 1200.0},
        point_feature_arrays=["wall_mask"],
        target_field_arrays=["velocity"],
        require_target_fields=True,
    )

    assert graph.y.shape == (4, 3)
    assert graph.target_field_names == ["velocity.x", "velocity.y", "velocity.z"]
