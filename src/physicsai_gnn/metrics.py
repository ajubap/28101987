import numpy as np

def array_metrics(pred, true):
    p = pred.detach().cpu().numpy().reshape(-1)
    t = true.detach().cpu().numpy().reshape(-1)

    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    mape = float(np.mean(np.abs((p - t) / np.maximum(np.abs(t), 1e-8))) * 100.0)

    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - np.mean(t)) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot

    return {"MAE": mae, "RMSE": rmse, "MAPE_%": mape, "R2": r2}


def field_metrics(pred, true, names):
    width = int(pred.shape[-1])
    names = list(names)
    if len(names) != width:
        names = names + [f"field_{i}" for i in range(len(names), width)]
    return {name: array_metrics(pred[:, i], true[:, i]) for i, name in enumerate(names[:width])}


def scalar_metrics(pred, true, names):
    width = int(pred.shape[-1])
    names = list(names)
    if len(names) != width:
        names = names + [f"scalar_{i}" for i in range(len(names), width)]
    return {name: array_metrics(pred[:, i], true[:, i]) for i, name in enumerate(names[:width])}
