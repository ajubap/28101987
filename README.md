# General CFD Surrogate Workbench

Train **MeshGNN surrogate models** for *any* CFD problem — not just one fixed
case — and drive the whole workflow from a clean Plotly Dash dashboard.

The model is a `UnifiedHybridCAEModel`: local MeshGraphNet message passing + a
global transformer context block + field and scalar (KPI) decoders, with
optional physics-aware losses. Everything about a problem — its global inputs,
per-node features, target fields, and scalar targets — is declared in a YAML
config, so the same engine handles a cold plate, an external aero body, or your
own case without code changes.

## What's in the box

```
configs/                YAML problem definitions (the contract for each case)
  generic_cfd_template.yaml   default, general-purpose template
  coldplate_config.yaml       worked example (thermal cold plate)
src/physicsai_gnn/      the engine: model, dataset, mesh→graph, training, inference
scripts/                command-line tools for every pipeline step
dashboard/              Plotly Dash multi-page UI (a thin layer over the scripts)
tests/                  unit tests
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .            # makes the physicsai_gnn package importable
```

`torch`, `torch-geometric`, `pyvista`, and `vtk` are heavyweight; install the
build that matches your platform/CUDA if the default wheels don't fit.

## Run the dashboard

```bash
python run_dashboard.py        # open http://127.0.0.1:8050
```

Pick the active config in the top-right dropdown. Every page reacts to it.

## The workflow (dashboard pages)

1. **Overview** – the active problem contract and model summary.
2. **Data** – generate a synthetic dataset, upload `.vtu/.vtk`, or convert an
   EnSight export, registering each case with its CFD inputs.
3. **Dataset** – validate the manifest and per-mesh arrays against the config.
4. **Train** – preprocess meshes into graphs, train, and watch the loss curve.
5. **Evaluate** – score a split and generate a validation report.
6. **Inference** – predict fields/KPIs on a new mesh, with optional MC-dropout
   uncertainty, and export an annotated VTU.
7. **Active Learning** – rank candidate cases by predictive uncertainty.
8. **Registry & Outputs** – quality gate, model versioning, artifact browser.

## Quick start (synthetic, any config)

```bash
# Build a learnable demo dataset that matches whatever the config declares
python scripts/generate_synthetic_dataset.py --config configs/generic_cfd_template.yaml --num_cases 60
python scripts/preprocess_cases.py           --config configs/generic_cfd_template.yaml
python scripts/train_meshgnn.py              --config configs/generic_cfd_template.yaml
python scripts/evaluate_meshgnn.py           --config configs/generic_cfd_template.yaml --split test
```

The same commands work for `configs/coldplate_config.yaml` (the cold-plate
example also has a bespoke generator, `generate_synthetic_coldplate.py`).

## Define your own problem

Copy `configs/generic_cfd_template.yaml` and edit the `data:` block:

```yaml
data:
  global_feature_columns: [reynolds, mach, angle_of_attack_deg]  # scalar inputs in cases.csv
  point_feature_arrays:   [wall_mask, inlet_mask, outlet_mask]   # per-node arrays in each mesh
  target_field_arrays:    [pressure, velocity]                   # fields to predict (velocity → 3 comps)
  target_scalar_columns:  [drag_coefficient, lift_coefficient]   # KPIs in cases.csv
```

Then point the dashboard (or the scripts) at your config, register meshes whose
`point_data` contains the declared arrays, and train. No code changes required.

> ⚠️ Surrogates are design-space approximations. Validate against full CFD/FEA
> before any engineering decision.
