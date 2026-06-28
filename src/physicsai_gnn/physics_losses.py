import torch
import torch.nn.functional as F

def smoothness_loss(pred_field, edge_index):
    src,dst=edge_index; return (pred_field[src]-pred_field[dst]).pow(2).mean()

def _per_graph_max(values,batch=None):
    if batch is None: return values.max().view(1)
    return torch.stack([values[batch==gid].max() for gid in torch.unique(batch)])

def _masked_mean(values,mask): return None if mask.sum()==0 else values[mask].mean()

def pressure_drop_positive_loss(pred_scalar, scalar_names):
    if pred_scalar is None or not scalar_names: return torch.tensor(0.0, device=pred_scalar.device if pred_scalar is not None else 'cpu')
    loss=pred_scalar.new_tensor(0.0)
    for i,name in enumerate(scalar_names):
        lname=name.lower()
        if 'pressure_drop' in lname or 'delta_p' in lname or 'dp' in lname:
            loss = loss + F.relu(-pred_scalar[:,i]).pow(2).mean()
    return loss

def _bc_col(cfg, bc_name):
    names=[n.lower() for n in cfg['data'].get('point_feature_arrays', [])]
    if bc_name.lower() not in names: return None
    return 3 + len(cfg['data'].get('global_feature_columns', [])) + names.index(bc_name.lower())

def boundary_condition_loss(pred_field,true_field,data,cfg):
    fields=[n.lower() for n in cfg['data'].get('target_field_arrays', [])]
    if 'temperature' not in fields: return pred_field.new_tensor(0.0)
    temp_idx=fields.index('temperature'); bc_col=_bc_col(cfg,'bc_inlet')
    if bc_col is None or data.x.shape[1] <= bc_col: return pred_field.new_tensor(0.0)
    mask=data.x[:,bc_col] > 0.5
    if mask.sum()==0: return pred_field.new_tensor(0.0)
    return F.mse_loss(pred_field[mask,temp_idx], true_field[mask,temp_idx])

def hotspot_loss(pred_field,true_field,data,cfg):
    fields=[n.lower() for n in cfg['data'].get('target_field_arrays', [])]
    if 'temperature' not in fields: return pred_field.new_tensor(0.0)
    idx=fields.index('temperature'); batch=getattr(data,'batch',None)
    return F.mse_loss(_per_graph_max(pred_field[:,idx],batch), _per_graph_max(true_field[:,idx],batch))

def pressure_drop_field_loss(pred_field,true_field,data,cfg):
    fields=[n.lower() for n in cfg['data'].get('target_field_arrays', [])]
    if 'pressure' not in fields: return pred_field.new_tensor(0.0)
    p_idx=fields.index('pressure'); inlet_col=_bc_col(cfg,'bc_inlet'); outlet_col=_bc_col(cfg,'bc_outlet')
    if inlet_col is None or outlet_col is None or data.x.shape[1] <= max(inlet_col,outlet_col): return pred_field.new_tensor(0.0)
    batch=getattr(data,'batch',None); losses=[]
    gids=[None] if batch is None else list(torch.unique(batch))
    for gid in gids:
        gmask=torch.ones(data.x.shape[0], dtype=torch.bool, device=data.x.device) if gid is None else (batch==gid)
        inlet=gmask & (data.x[:,inlet_col] > 0.5); outlet=gmask & (data.x[:,outlet_col] > 0.5)
        pinp=_masked_mean(pred_field[:,p_idx], inlet); poutp=_masked_mean(pred_field[:,p_idx], outlet)
        pint=_masked_mean(true_field[:,p_idx], inlet); poutt=_masked_mean(true_field[:,p_idx], outlet)
        if any(v is None for v in [pinp,poutp,pint,poutt]): continue
        losses.append(((pinp-poutp)-(pint-poutt)).pow(2))
    return torch.stack(losses).mean() if losses else pred_field.new_tensor(0.0)

def combined_loss(outputs,data,cfg):
    lc=cfg.get('loss', {})
    loss=float(lc.get('field_weight',1.0))*F.mse_loss(outputs['field'], data.y)
    if 'scalar' in outputs and hasattr(data,'y_scalar'):
        loss = loss + float(lc.get('scalar_weight',0.3))*F.mse_loss(outputs['scalar'], data.y_scalar)
        loss = loss + float(lc.get('pressure_positive_weight',0.0))*pressure_drop_positive_loss(outputs['scalar'], cfg['data'].get('target_scalar_columns', []))
    if float(lc.get('smoothness_weight',0.0))>0: loss = loss + float(lc.get('smoothness_weight',0.0))*smoothness_loss(outputs['field'], data.edge_index)
    if float(lc.get('boundary_condition_weight',0.0))>0: loss = loss + float(lc.get('boundary_condition_weight',0.0))*boundary_condition_loss(outputs['field'], data.y, data, cfg)
    if float(lc.get('hotspot_weight',0.0))>0: loss = loss + float(lc.get('hotspot_weight',0.0))*hotspot_loss(outputs['field'], data.y, data, cfg)
    if float(lc.get('pressure_drop_field_weight',0.0))>0: loss = loss + float(lc.get('pressure_drop_field_weight',0.0))*pressure_drop_field_loss(outputs['field'], data.y, data, cfg)
    return loss
