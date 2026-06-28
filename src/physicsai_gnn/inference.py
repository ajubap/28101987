import torch

from .normalization import apply_normalizers, inverse_field, inverse_scalar


def enable_dropout(model):
    for module in model.modules():
        if module.__class__.__name__.lower().startswith("dropout"):
            module.train()


@torch.no_grad()
def predict_once(model, data, device, normalizers=None):
    model.eval()
    d = apply_normalizers(data, normalizers).to(device)
    out = model(d)

    field = inverse_field(out["field"], normalizers).detach().cpu()
    scalar = None
    if "scalar" in out:
        scalar = inverse_scalar(out["scalar"], normalizers).detach().cpu()

    return field, scalar


@torch.no_grad()
def mc_dropout_predict(model, data, device, normalizers=None, samples=20):
    model.train()
    enable_dropout(model)

    fields = []
    scalars = []

    for _ in range(samples):
        d = apply_normalizers(data, normalizers).to(device)
        out = model(d)

        field = inverse_field(out["field"], normalizers).detach().cpu()
        fields.append(field)

        if "scalar" in out:
            scalar = inverse_scalar(out["scalar"], normalizers).detach().cpu()
            scalars.append(scalar)

    field_stack = torch.stack(fields, dim=0)
    field_mean = field_stack.mean(dim=0)
    field_std = field_stack.std(dim=0)

    scalar_mean = None
    scalar_std = None
    if scalars:
        scalar_stack = torch.stack(scalars, dim=0)
        scalar_mean = scalar_stack.mean(dim=0)
        scalar_std = scalar_stack.std(dim=0)

    return field_mean, field_std, scalar_mean, scalar_std
