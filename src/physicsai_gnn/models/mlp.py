from torch import nn

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, layers=3, dropout=0.0, layer_norm=True):
        super().__init__()
        mods = []
        d = in_dim

        for _ in range(max(layers - 1, 1)):
            mods.append(nn.Linear(d, hidden_dim))
            mods.append(nn.SiLU())
            if dropout > 0:
                mods.append(nn.Dropout(dropout))
            d = hidden_dim

        mods.append(nn.Linear(d, out_dim))
        if layer_norm:
            mods.append(nn.LayerNorm(out_dim))

        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)
