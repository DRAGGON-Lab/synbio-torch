"""A graph transformer over SBOL composition graphs (PyG TransformerConv)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, global_mean_pool

from synbiotorch.config import GraphArch, ModelConfig, TaskConfig
from synbiotorch.encoders.graph import GraphSpec
from synbiotorch.exceptions import ConfigError
from synbiotorch.models.heads import build_head


class GraphTransformerModel(nn.Module):
    """Embeds node (class, role) and edge types, applies attention convs, pools, heads."""

    def __init__(
        self,
        spec: GraphSpec,
        *,
        hidden_size: int,
        num_layers: int,
        attn_heads: int,
        head: nn.Module,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_size % attn_heads != 0:
            raise ConfigError(f"hidden_size {hidden_size} must be divisible by attn_heads {attn_heads}")
        self.class_emb = nn.Embedding(spec.num_node_classes, hidden_size)
        self.role_emb = nn.Embedding(spec.num_roles, hidden_size)
        self.name_emb = nn.Embedding(spec.num_name_buckets, hidden_size)
        self.edge_emb = nn.Embedding(spec.num_edge_types, hidden_size)
        self.convs = nn.ModuleList(
            TransformerConv(
                hidden_size, hidden_size // attn_heads, heads=attn_heads, edge_dim=hidden_size, dropout=dropout
            )
            for _ in range(num_layers)
        )
        self.head = head

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        h = self.class_emb(x[:, 0]) + self.role_emb(x[:, 1]) + self.name_emb(x[:, 2])
        edge_attr = self.edge_emb(edge_type)
        for conv in self.convs:
            h = torch.relu(conv(h, edge_index, edge_attr))
        pooled = global_mean_pool(h, batch)
        return self.head(pooled)


def build_graph_model(model_config: ModelConfig, task_config: TaskConfig, spec: GraphSpec) -> GraphTransformerModel:
    arch = model_config.arch
    if not isinstance(arch, GraphArch):
        raise ConfigError(f"the graph model requires model.arch.family 'graph', not '{arch.family}'")
    hidden = model_config.hidden_size
    head = build_head(task_config, hidden, model_config.dropout)
    return GraphTransformerModel(
        spec,
        hidden_size=hidden,
        num_layers=arch.num_hidden_layers,
        attn_heads=arch.num_attention_heads,
        head=head,
        dropout=model_config.dropout,
    )
