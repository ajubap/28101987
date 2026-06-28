from pathlib import Path
import json
import torch

class TensorNormalizer:
    def __init__(self, mean=None, std=None, eps=1e-8):
        self.mean = mean; self.std = std; self.eps = eps
    def fit(self, tensors):
        flat=[]
        for t in tensors:
            if t is None: continue
            x=t.detach().float(); x=x.view(-1,1) if x.ndim==1 else x.reshape(-1,x.shape[-1]); flat.append(x)
        if not flat: raise ValueError('No tensors provided for normalizer fitting.')
        cat=torch.cat(flat, dim=0)
        self.mean=cat.mean(dim=0)
        self.std=cat.std(dim=0, unbiased=False)
        self.std=torch.nan_to_num(self.std, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(self.eps)
        return self
    def set_identity_dims(self, dims):
        if self.mean is not None and self.std is not None:
            for d in dims: self.mean[d]=0.0; self.std[d]=1.0
        return self
    def transform(self,x): return (x-self.mean.to(x.device))/self.std.to(x.device)
    def inverse(self,x): return x*self.std.to(x.device)+self.mean.to(x.device)
    def state_dict(self): return {'mean': self.mean.detach().cpu().tolist(), 'std': self.std.detach().cpu().tolist(), 'eps': self.eps}
    @classmethod
    def from_state_dict(cls,state): return cls(torch.tensor(state['mean'], dtype=torch.float32), torch.tensor(state['std'], dtype=torch.float32), float(state.get('eps',1e-8)))

def _x_identity_dims(dataset, cfg):
    pats=[p.lower() for p in cfg.get('normalization',{}).get('no_normalize_x_patterns', [])]
    if not pats: return []
    names=list(getattr(dataset[0], 'x_feature_names', [])); dims=[]
    for i,name in enumerate(names):
        if any(p in str(name).lower() for p in pats): dims.append(i)
    return dims

def fit_normalizers(dataset, cfg):
    nc=cfg.get('normalization', {}); norms={}
    if not nc.get('enabled', True): return norms
    xs=[]; es=[]; ys=[]; scalars=[]
    for data in dataset:
        if nc.get('normalize_x', True): xs.append(data.x)
        if nc.get('normalize_edge_attr', True): es.append(data.edge_attr)
        if nc.get('normalize_y', True): ys.append(data.y)
        if nc.get('normalize_y_scalar', True) and hasattr(data, 'y_scalar'): scalars.append(data.y_scalar)
    if xs: norms['x']=TensorNormalizer().fit(xs).set_identity_dims(_x_identity_dims(dataset,cfg))
    if es: norms['edge_attr']=TensorNormalizer().fit(es)
    if ys: norms['y']=TensorNormalizer().fit(ys)
    if scalars: norms['y_scalar']=TensorNormalizer().fit(scalars)
    return norms

def apply_normalizers(data, normalizers):
    if not normalizers: return data
    data=data.clone()
    if 'x' in normalizers: data.x=normalizers['x'].transform(data.x)
    if 'edge_attr' in normalizers: data.edge_attr=normalizers['edge_attr'].transform(data.edge_attr)
    if 'y' in normalizers and hasattr(data,'y'): data.y=normalizers['y'].transform(data.y)
    if 'y_scalar' in normalizers and hasattr(data,'y_scalar'): data.y_scalar=normalizers['y_scalar'].transform(data.y_scalar)
    return data

def inverse_field(pred_field, normalizers): return normalizers['y'].inverse(pred_field) if normalizers and 'y' in normalizers else pred_field
def inverse_scalar(pred_scalar, normalizers): return normalizers['y_scalar'].inverse(pred_scalar) if normalizers and 'y_scalar' in normalizers else pred_scalar

def save_normalizers(path, normalizers):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k:v.state_dict() for k,v in normalizers.items()}, indent=2), encoding='utf-8')

def load_normalizers(path):
    path=Path(path)
    if not path.exists(): return {}
    payload=json.loads(path.read_text(encoding='utf-8'))
    return {k: TensorNormalizer.from_state_dict(v) for k,v in payload.items()}
