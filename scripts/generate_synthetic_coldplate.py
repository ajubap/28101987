import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import meshio


def make_grid(nx, ny, length, width):
    xs = np.linspace(0, length, nx)
    ys = np.linspace(0, width, ny)
    points = np.array([[x, y, 0.0] for y in ys for x in xs], dtype=np.float32)

    quads = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            quads.append([n0, n0 + 1, n0 + nx + 1, n0 + nx])

    return points, np.array(quads, dtype=np.int64)


def synthetic_fields(points, flow, inlet, power, thickness):
    x = points[:, 0]
    y = points[:, 1]
    L = x.max()
    W = y.max()

    heat = np.exp(-((x - 0.55 * L) ** 2) / (2 * (0.15 * L) ** 2))
    heat *= np.exp(-((y - 0.50 * W) ** 2) / (2 * (0.25 * W) ** 2))

    temp = inlet + power * 0.55 * (1.0 / (0.4 + flow)) * (6.0 / thickness) * (0.25 + heat)
    temp += 1.5 * np.sin(2 * np.pi * x / L) * np.cos(np.pi * y / W)

    dp = 9000.0 * flow ** 1.8 * (6.0 / thickness) ** 0.4
    pressure = dp * (1.0 - x / L) + 80.0 * np.sin(np.pi * y / W)

    return temp.astype(np.float32), pressure.astype(np.float32), float(dp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cases", type=int, default=40)
    parser.add_argument("--out_dir", default="data/raw")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    rows = []

    for k in range(args.num_cases):
        case_id = f"case_{k+1:04d}"

        nx = int(rng.integers(36, 56))
        ny = int(rng.integers(14, 24))
        length = float(rng.uniform(90, 115))
        width = float(rng.uniform(24, 36))

        flow = float(rng.uniform(0.3, 2.0))
        inlet = float(rng.uniform(20, 40))
        power = float(rng.choice([16, 24, 32, 40]))
        thickness = float(rng.uniform(4.5, 9.0))

        points, quads = make_grid(nx, ny, length, width)
        temp, pressure, dp = synthetic_fields(points, flow, inlet, power, thickness)

        x = points[:, 0]
        y = points[:, 1]
        L = x.max()
        W = y.max()

        bc_inlet = (x < 0.02 * L).astype(np.float32)
        bc_outlet = (x > 0.98 * L).astype(np.float32)
        bc_wall = ((y < 0.02 * W) | (y > 0.98 * W)).astype(np.float32)
        bc_heat = (((x > 0.38 * L) & (x < 0.72 * L) & (y > 0.30 * W) & (y < 0.70 * W))).astype(np.float32)

        mesh = meshio.Mesh(
            points=points,
            cells=[("quad", quads)],
            point_data={
                "temperature": temp,
                "pressure": pressure,
                "bc_inlet": bc_inlet,
                "bc_outlet": bc_outlet,
                "bc_wall": bc_wall,
                "bc_heat_source": bc_heat,
            },
        )

        mesh_file = f"{case_id}.vtu"
        meshio.write(out_dir / mesh_file, mesh)

        rows.append({
            "case_id": case_id,
            "mesh_file": mesh_file,
            "flow_rate_lpm": flow,
            "inlet_temp_c": inlet,
            "power_w": power,
            "plate_thickness_mm": thickness,
            "target_max_temp_c": float(temp.max()),
            "target_pressure_drop_pa": dp,
        })

    pd.DataFrame(rows).to_csv(out_dir / "cases.csv", index=False)
    print(f"Wrote {len(rows)} synthetic cases to {out_dir}")


if __name__ == "__main__":
    main()
