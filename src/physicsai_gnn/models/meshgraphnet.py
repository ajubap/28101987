import torch
from torch import nn

try:
    from torch_scatter import scatter_mean
except Exception:
    scatter_mean = None

from .mlp import MLP


def scatter_mean_safe(src, index, dim_size):
    if scatter_mean is not None:
        return scatter_mean(src, index, dim=0, dim_size=dim_size)

    out = src.new_zeros((dim_size, src.shape[-1]))
    count = src.new_zeros((dim_size, 1))
    out.index_add_(0, index, src)
    count.index_add_(0, index, torch.ones((src.shape[0], 1), dtype=src.dtype, device=src.device))
    return out / count.clamp_min(1.0)


class MeshGraphNetBlock(nn.Module):
    def __init__(self, latent_dim, mlp_layers=3, dropout=0.0, layer_norm=True):
        super().__init__()

        self.edge_mlp = MLP(
            in_dim=latent_dim * 3,
            out_dim=latent_dim,
            hidden_dim=latent_dim,
            layers=mlp_layers,
            dropout=dropout,
            layer_norm=layer_norm,
        )

        self.node_mlp = MLP(
            in_dim=latent_dim * 2,
            out_dim=latent_dim,
            hidden_dim=latent_dim,
            layers=mlp_layers,
            dropout=dropout,
            layer_norm=layer_norm,
        )

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index

        edge_input = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        edge_update = self.edge_mlp(edge_input)
        edge_attr = edge_attr + edge_update

        aggregated = scatter_mean_safe(edge_attr, dst, dim_size=x.shape[0])
        node_input = torch.cat([x, aggregated], dim=-1)
        node_update = self.node_mlp(node_input)
        x = x + node_update

        return x, edge_attr


class MeshGraphNet(nn.Module):
    def __init__(
        self,
        node_in_dim,
        edge_in_dim,
        field_out_dim,
        scalar_out_dim=0,
        latent_dim=128,
        message_passing_steps=10,
        mlp_layers=3,
        dropout=0.0,
        use_layer_norm=True,
        predict_scalar_outputs=True,
    ):
        super().__init__()

        self.predict_scalar_outputs = bool(predict_scalar_outputs and scalar_out_dim > 0)

        self.node_encoder = MLP(node_in_dim, latent_dim, latent_dim, mlp_layers, dropout, use_layer_norm)
        self.edge_encoder = MLP(edge_in_dim, latent_dim, latent_dim, mlp_layers, dropout, use_layer_norm)

        self.processor = nn.ModuleList([
            MeshGraphNetBlock(latent_dim, mlp_layers, dropout, use_layer_norm)
            for _ in range(message_passing_steps)
        ])

        self.field_decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, field_out_dim),
        )

        if self.predict_scalar_outputs:
            self.scalar_decoder = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.SiLU(),
                nn.Linear(latent_dim, scalar_out_dim),
            )

    def forward(self, data):
        x = self.node_encoder(data.x)
        e = self.edge_encoder(data.edge_attr)

        for block in self.processor:
            x, e = block(x, data.edge_index, e)

        out = {"field": self.field_decoder(x)}

        if self.predict_scalar_outputs:
            if hasattr(data, "batch"):
                batch = data.batch
                pooled = scatter_mean_safe(x, batch, int(batch.max().item()) + 1)
            else:
                pooled = x.mean(dim=0, keepdim=True)

            out["scalar"] = self.scalar_decoder(pooled)

        return out
