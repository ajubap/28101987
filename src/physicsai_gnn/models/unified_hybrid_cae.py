import torch
from torch import nn
from .mlp import MLP
from .meshgraphnet import MeshGraphNetBlock, scatter_mean_safe

class GraphGlobalTransformerBlock(nn.Module):
    def __init__(self, latent_dim, heads=4, layers=1, dropout=0.0, max_tokens=512, mlp_layers=3, layer_norm=True):
        super().__init__()
        self.max_tokens = int(max_tokens)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=heads, dim_feedforward=latent_dim*4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.context_mlp = MLP(latent_dim*2, latent_dim, latent_dim, mlp_layers, dropout, layer_norm)
    def _sample_tokens(self, x_graph):
        n = x_graph.shape[0]
        if n <= self.max_tokens:
            return x_graph
        idx = torch.linspace(0, n-1, self.max_tokens, device=x_graph.device).long()
        return x_graph[idx]
    def forward(self, x, batch=None):
        if batch is None:
            tokens = self._sample_tokens(x).unsqueeze(0)
            transformed = self.transformer(tokens).squeeze(0)
            context = transformed.mean(dim=0, keepdim=True).repeat(x.shape[0], 1)
            return x + self.context_mlp(torch.cat([x, context], dim=-1))
        out = torch.empty_like(x)
        for gid in torch.unique(batch):
            mask = batch == gid
            xg = x[mask]
            tokens = self._sample_tokens(xg).unsqueeze(0)
            transformed = self.transformer(tokens).squeeze(0)
            context = transformed.mean(dim=0, keepdim=True).repeat(xg.shape[0], 1)
            out[mask] = xg + self.context_mlp(torch.cat([xg, context], dim=-1))
        return out

class UnifiedHybridCAEModel(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, field_out_dim, scalar_out_dim=0,
                 latent_dim=160, local_message_passing_steps=8, global_transformer_steps=2,
                 transformer_heads=4, transformer_layers_per_step=1, max_global_tokens=512,
                 mlp_layers=3, dropout=0.08, use_layer_norm=True, predict_scalar_outputs=True):
        super().__init__()
        self.predict_scalar_outputs = bool(predict_scalar_outputs and scalar_out_dim > 0)
        self.node_encoder = MLP(node_in_dim, latent_dim, latent_dim, mlp_layers, dropout, use_layer_norm)
        self.edge_encoder = MLP(edge_in_dim, latent_dim, latent_dim, mlp_layers, dropout, use_layer_norm)
        self.local_processor = nn.ModuleList([
            MeshGraphNetBlock(latent_dim, mlp_layers=mlp_layers, dropout=dropout, layer_norm=use_layer_norm)
            for _ in range(local_message_passing_steps)
        ])
        self.global_processor = nn.ModuleList([
            GraphGlobalTransformerBlock(latent_dim, transformer_heads, transformer_layers_per_step, dropout,
                                        max_global_tokens, mlp_layers, use_layer_norm)
            for _ in range(global_transformer_steps)
        ])
        self.field_decoder = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.SiLU(), nn.Dropout(dropout),
                                           nn.Linear(latent_dim, latent_dim//2), nn.SiLU(),
                                           nn.Linear(latent_dim//2, field_out_dim))
        if self.predict_scalar_outputs:
            self.scalar_decoder = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.SiLU(), nn.Dropout(dropout),
                                                nn.Linear(latent_dim, latent_dim//2), nn.SiLU(),
                                                nn.Linear(latent_dim//2, scalar_out_dim))
    def forward(self, data):
        x = self.node_encoder(data.x)
        e = self.edge_encoder(data.edge_attr)
        for block in self.local_processor:
            x, e = block(x, data.edge_index, e)
        batch = getattr(data, 'batch', None)
        for block in self.global_processor:
            x = block(x, batch=batch)
        out = {'field': self.field_decoder(x)}
        if self.predict_scalar_outputs:
            if batch is not None:
                pooled = scatter_mean_safe(x, batch, int(batch.max().item())+1)
            else:
                pooled = x.mean(dim=0, keepdim=True)
            out['scalar'] = self.scalar_decoder(pooled)
        return out
