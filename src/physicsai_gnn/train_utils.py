from pathlib import Path
import random
import numpy as np
import torch
from .models.meshgraphnet import MeshGraphNet
from .models.unified_hybrid_cae import UnifiedHybridCAEModel
from .physics_losses import combined_loss
from .normalization import apply_normalizers

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def choose_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)

def infer_dims(dataset):
    sample = dataset[0]
    node_in = sample.x.shape[-1]; edge_in = sample.edge_attr.shape[-1]; field_out = sample.y.shape[-1]
    scalar_out = sample.y_scalar.shape[-1] if hasattr(sample, 'y_scalar') else 0
    return node_in, edge_in, field_out, scalar_out

def build_model(cfg, dataset):
    node_in, edge_in, field_out, scalar_out = infer_dims(dataset)
    mc = cfg['model']; name = str(mc.get('name', 'meshgraphnet')).lower()
    if name == 'unified_hybrid_cae':
        return UnifiedHybridCAEModel(
            node_in_dim=node_in, edge_in_dim=edge_in, field_out_dim=field_out, scalar_out_dim=scalar_out,
            latent_dim=int(mc.get('latent_dim', 160)),
            local_message_passing_steps=int(mc.get('local_message_passing_steps', 8)),
            global_transformer_steps=int(mc.get('global_transformer_steps', 2)),
            transformer_heads=int(mc.get('transformer_heads', 4)),
            transformer_layers_per_step=int(mc.get('transformer_layers_per_step', 1)),
            max_global_tokens=int(mc.get('max_global_tokens', 512)),
            mlp_layers=int(mc.get('mlp_layers', 3)), dropout=float(mc.get('dropout', 0.08)),
            use_layer_norm=bool(mc.get('use_layer_norm', True)),
            predict_scalar_outputs=bool(mc.get('predict_scalar_outputs', True)))
    return MeshGraphNet(node_in, edge_in, field_out, scalar_out, int(mc.get('latent_dim', 128)),
                        int(mc.get('message_passing_steps', 10)), int(mc.get('mlp_layers', 3)),
                        float(mc.get('dropout', 0.0)), bool(mc.get('use_layer_norm', True)),
                        bool(mc.get('predict_scalar_outputs', True)))

def train_one_epoch(model, loader, optimizer, cfg, device, normalizers=None):
    model.train(); total=0.0; n=0
    for data in loader:
        data = apply_normalizers(data, normalizers).to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(data)
        loss = combined_loss(outputs, data, cfg)
        loss.backward()
        clip = float(cfg['training'].get('grad_clip_norm', 0.0))
        if clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step(); total += float(loss.detach().cpu()); n += 1
    return total / max(n, 1)

@torch.no_grad()
def evaluate_loss(model, loader, cfg, device, normalizers=None):
    model.eval(); total=0.0; n=0
    for data in loader:
        data = apply_normalizers(data, normalizers).to(device)
        outputs = model(data)
        loss = combined_loss(outputs, data, cfg)
        total += float(loss.detach().cpu()); n += 1
    return total / max(n, 1)

def save_checkpoint(path, model, cfg, epoch, best_val):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'config': cfg, 'epoch': epoch, 'best_val': best_val}, path)

def load_checkpoint(path, model, map_location='cpu'):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    return ckpt
