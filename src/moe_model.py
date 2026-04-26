"""MoE with Graph-Conditioned Routing and ST-LoRA Adapters."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.linalg import eigsh as sp_eigsh
import scipy.sparse as _scipy_sparse
from .model import Node_Specific_Predictor, GCN

_GRAPH_FEATURE_CACHE: dict = {}

def _compute_graph_features_cached(adj_mx, node_num: int) -> 'torch.Tensor':
    """Compute 9 normalized topology features [N, 9], cached by adj_mx id."""
    import networkx as nx
    cache_key = id(adj_mx)
    if cache_key in _GRAPH_FEATURE_CACHE:
        return _GRAPH_FEATURE_CACHE[cache_key]

    if isinstance(adj_mx, torch.Tensor):
        adj_np = adj_mx.cpu().numpy()
    elif isinstance(adj_mx, list):
        adj_np = adj_mx[0].cpu().numpy() if isinstance(adj_mx[0], torch.Tensor) \
                 else np.array(adj_mx[0])
    else:
        adj_np = np.array(adj_mx)
    if adj_np.shape[0] != node_num:
        adj_np = adj_np[:node_num, :node_num]

    G = nx.from_numpy_array(adj_np)
    feats = []

    def _norm(arr):
        arr = np.array(arr, dtype=np.float32)
        rng = arr.max() - arr.min()
        return (arr - arr.min()) / (rng + 1e-8)

    degree = np.array([G.degree(i, weight='weight') for i in range(node_num)], dtype=np.float32)
    feats.append(_norm(degree))

    try:
        close_dict = nx.closeness_centrality(G)
        close = np.array([close_dict[i] for i in range(node_num)], dtype=np.float32)
    except Exception:
        close = np.zeros(node_num, dtype=np.float32)
    feats.append(_norm(close))

    try:
        clust_dict = nx.clustering(G, weight='weight')
        clust = np.array([clust_dict[i] for i in range(node_num)], dtype=np.float32)
    except Exception:
        clust = np.zeros(node_num, dtype=np.float32)
    feats.append(_norm(clust))

    try:
        pr = nx.pagerank(G, weight='weight')
        pr_arr = np.array([pr[i] for i in range(node_num)], dtype=np.float32)
    except Exception:
        pr_arr = np.ones(node_num, dtype=np.float32) / node_num
    feats.append(_norm(pr_arr))

    try:
        btwn = nx.betweenness_centrality(G, weight='weight', normalized=True)
        btwn_arr = np.array([btwn[i] for i in range(node_num)], dtype=np.float32)
    except Exception:
        btwn_arr = np.zeros(node_num, dtype=np.float32)
    feats.append(_norm(btwn_arr))

    try:
        G_simple = nx.Graph(G)          # k_core requires a simple Graph
        core_dict = nx.core_number(G_simple)
        core_arr = np.array([core_dict[i] for i in range(node_num)], dtype=np.float32)
    except Exception:
        core_arr = np.zeros(node_num, dtype=np.float32)
    feats.append(_norm(core_arr))

    try:
        ec = nx.eigenvector_centrality(G, max_iter=1000, weight='weight')
        ec_arr = np.array([ec[i] for i in range(node_num)], dtype=np.float32)
    except Exception:
        ec_arr = np.zeros(node_num, dtype=np.float32)
    feats.append(_norm(ec_arr))

    # Spectral features: Fiedler vector + 3rd Laplacian eigenvector
    try:
        L = nx.normalized_laplacian_matrix(G, weight='weight').astype(np.float64)
        k = min(4, node_num - 1)
        vals, vecs = sp_eigsh(L, k=k, which='SM', tol=1e-5)
        order = np.argsort(vals)
        vecs = vecs[:, order]
        fiedler = vecs[:, 1].astype(np.float32) if k >= 2 else np.zeros(node_num, dtype=np.float32)
        spec3   = vecs[:, 2].astype(np.float32) if k >= 3 else np.zeros(node_num, dtype=np.float32)
    except Exception:
        fiedler = np.zeros(node_num, dtype=np.float32)
        spec3   = np.zeros(node_num, dtype=np.float32)
    feats.append(_norm(fiedler))
    feats.append(_norm(spec3))

    result = torch.FloatTensor(np.stack(feats, axis=1))  # [N, 9]
    _GRAPH_FEATURE_CACHE[cache_key] = result
    return result


class NodeInputProjection(nn.Module):
    """Per-expert, graph-conditioned input projection with temporal attention."""

    def __init__(self, num_experts, node_num, input_dim, adj_mx,
                 proj_hidden_dim=32, shared_projection=False, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.node_num = node_num
        self.input_dim = input_dim
        self.shared_projection = shared_projection

        # Static graph features (shared cache)
        graph_feats = _compute_graph_features_cached(adj_mx, node_num)
        self.register_buffer('graph_feats', graph_feats)
        graph_feat_dim = graph_feats.shape[1]

        # Dynamic input encoder
        dynamic_dim = 16
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, dynamic_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Temporal attention: learn which timesteps matter for projection
        self.temporal_attn = nn.Linear(dynamic_dim, 1)
        self.dynamic_dim = dynamic_dim

        # Combined conditioning
        combined_dim = graph_feat_dim + dynamic_dim

        num_projections = 1 if shared_projection else num_experts

        # Hypernetwork: combined features -> per-node gating
        self.hyper_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(combined_dim, proj_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(proj_hidden_dim, input_dim),
                nn.Sigmoid(),
            ) for _ in range(num_projections)
        ])

        # Per-expert global learned transform
        self.proj_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim, input_dim),
            ) for _ in range(num_projections)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(input_dim) for _ in range(num_projections)
        ])

        # Startup gate: sigmoid(-3) ~ 0.05 -> near identity at init
        self.proj_gate = nn.Parameter(torch.full((num_projections,), -3.0))

        self.dropout = nn.Dropout(dropout)

        # Zero-init output layers -> residual starts clean
        for transform in self.proj_transforms:
            nn.init.zeros_(transform[-1].weight)
            if transform[-1].bias is not None:
                nn.init.zeros_(transform[-1].bias)

    # ------------------------------------------------------------------ #
    def _encode_input(self, x):
        """Temporal-attention pooling: [B, T, N, D] -> [B, N, dynamic_dim]."""
        h = self.input_encoder(x)
        attn_scores = self.temporal_attn(h)
        attn_weights = torch.softmax(attn_scores, dim=1)
        return (h * attn_weights).sum(dim=1)

    # ------------------------------------------------------------------ #
    def forward(self, x, expert_idx=None):
        """Apply per-expert input projections. Returns list or single [B,T,N,D]."""
        B, T, N, D = x.shape

        dynamic_feat = self._encode_input(x)
        static_feat = self.graph_feats.unsqueeze(0).expand(B, -1, -1)
        combined_feat = torch.cat([static_feat, dynamic_feat], dim=-1)

        if expert_idx is not None:
            idx = 0 if self.shared_projection else expert_idx
            return self._project_single(x, idx, combined_feat)
        else:
            return [self._project_single(x, 0 if self.shared_projection else i, combined_feat)
                    for i in range(self.num_experts)]

    def _project_single(self, x, proj_idx, combined_feat):
        """Apply a single expert's input-aware projection with gated residual."""
        # combined_feat: [B, N, combined_dim]
        node_scales = self.hyper_nets[proj_idx](combined_feat)
        projected = self.proj_transforms[proj_idx](x)
        delta = projected * node_scales.unsqueeze(1)
        gate = torch.sigmoid(self.proj_gate[proj_idx])
        return self.layer_norms[proj_idx](x + gate * self.dropout(delta))


class GraphTopologyFeatures:
    """Wrapper around the module-level graph feature cache."""

    def __init__(self, adj_mx, device):
        self.device = device
        if isinstance(adj_mx, torch.Tensor):
            adj_np = adj_mx.cpu().numpy()
        else:
            adj_np = np.array(adj_mx)
        N = adj_np.shape[0]

        feats = _compute_graph_features_cached(adj_mx if not isinstance(adj_mx, np.ndarray)
                                               else adj_mx, N)
        self.features = feats.to(device)

        self.degree_centrality    = self.features[:, 0]
        self.closeness_centrality = self.features[:, 1]
        self.clustering_coef      = self.features[:, 2]
        self.pagerank             = self.features[:, 3]
        self.betweenness          = self.features[:, 4] if self.features.shape[1] > 4 else None
        self.kcore                = self.features[:, 5] if self.features.shape[1] > 5 else None
        self.eigenvec_cent        = self.features[:, 6] if self.features.shape[1] > 6 else None
        self.fiedler              = self.features[:, 7] if self.features.shape[1] > 7 else None

    def get_features(self):
        return self.features


class GraphConditionedExpertRouter(nn.Module):
    """Routes nodes to experts using static graph topology + dynamic traffic input.

    Static pathway: graph features + learnable node embeddings + neighbour aggregation.
    Dynamic pathway: temporal attention over input + spatial context via adjacency.
    Adaptive gate blends both pathways per node per sample.
    """
    ROUTER_TYPE = 'graph_conditioned'

    def __init__(self, num_experts, node_num, graph_features,
                 embed_dim=32, use_input_context=True, temperature=1.0,
                 input_dim=None, top_k=None, noise_scale=0.5, early_routing=False,
                 routing_mode='soft', adj_mx=None):
        super().__init__()
        self.num_experts = num_experts
        self.node_num = node_num
        self.embed_dim = embed_dim
        self.use_input_context = use_input_context
        self.temperature = temperature
        self.early_routing = early_routing
        self.routing_mode = routing_mode

        self.top_k = max(1, min(top_k if top_k is not None else num_experts, num_experts))
        self.noise_scale = noise_scale

        self.register_buffer('graph_features', graph_features.get_features())
        num_graph_features = self.graph_features.shape[1]

        self.feature_projection = nn.Sequential(
            nn.Linear(num_graph_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Learnable node embedding
        self.node_embedding = nn.Parameter(torch.randn(node_num, embed_dim) * 0.01)

        # Static routing bias from graph topology
        self.static_routing = nn.Parameter(torch.randn(node_num, num_experts) * 0.01)

        # Neighbourhood aggregation for static pathway
        static_norm_adj = self._build_norm_adj(adj_mx, node_num)
        self.register_buffer('static_norm_adj', static_norm_adj)
        self.graph_neighbor_gate = nn.Parameter(torch.tensor(0.5))

        # Dynamic pathway: input-aware routing
        if use_input_context:
            if input_dim is None:
                raise ValueError("input_dim must be provided when use_input_context=True")

            dynamic_dim = 32

            self.input_encoder = nn.Sequential(
                nn.Linear(input_dim, dynamic_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
            # Temporal attention
            self.temporal_attn = nn.Sequential(
                nn.Linear(dynamic_dim, dynamic_dim // 2),
                nn.ReLU(),
                nn.Linear(dynamic_dim // 2, 1),
            )
            # Spatial context via normalized adjacency
            norm_adj = self._build_norm_adj(adj_mx, node_num)
            self.register_buffer('norm_adj', norm_adj)
            self.spatial_mixer = nn.Linear(dynamic_dim, dynamic_dim)

            # Project dynamic features to match embed_dim
            self.dynamic_proj = nn.Sequential(
                nn.Linear(dynamic_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )

            # Adaptive gate: static vs dynamic
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, 1),
                nn.Sigmoid(),
            )

            # Dynamic routing head
            self.dynamic_router = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim, num_experts),
            )
        else:
            # Fallback: input_projection for simple mean-pooling (backward compat)
            if input_dim is not None and input_dim != embed_dim:
                self.input_projection = nn.Linear(input_dim, embed_dim)
            else:
                self.input_projection = None

            self.input_encoder_simple = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim, num_experts),
            ) if input_dim is not None else None

        self._weights_initialized = False

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_norm_adj(adj_mx, node_num):
        """Row-normalize adjacency with self-loops."""
        if adj_mx is None:
            return torch.eye(node_num)
        if isinstance(adj_mx, torch.Tensor):
            adj = adj_mx.float()
        elif isinstance(adj_mx, list):
            adj = adj_mx[0].float() if isinstance(adj_mx[0], torch.Tensor) else torch.FloatTensor(adj_mx[0])
        else:
            adj = torch.FloatTensor(adj_mx)
        if adj.shape[0] != node_num:
            adj = adj[:node_num, :node_num]
        adj = adj + torch.eye(node_num)
        row_sum = adj.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return adj / row_sum

    # ------------------------------------------------------------------ #
    def _init_weights(self):
        """Initialize static routing near-uniform."""
        if self._weights_initialized:
            return
        with torch.no_grad():
            val = 1.0 / self.num_experts
            for i in range(self.num_experts):
                pref = torch.full((self.node_num,), val, device=self.static_routing.device)
                noise = torch.randn(self.node_num, device=self.static_routing.device) * (val * 0.25)
                self.static_routing[:, i] = pref + noise
            self._weights_initialized = True

    # ------------------------------------------------------------------ #
    def _noisy_topk(self, logits, k, training=True):
        """Noisy top-k gating (ST-MoE style)."""
        if k >= self.num_experts:
            return F.softmax(logits / self.temperature, dim=-1)
        if training:
            logits = logits + torch.randn_like(logits) * self.noise_scale
        topk_vals, topk_idx = logits.topk(k, dim=-1)
        masked = torch.full_like(logits, float('-inf'))
        masked.scatter_(-1, topk_idx, topk_vals)
        return F.softmax(masked / self.temperature, dim=-1)

    # ------------------------------------------------------------------ #
    def forward(self, x, return_weights=False):
        if not self._weights_initialized:
            self._init_weights()

        B, T, N, D = x.shape

        # Static pathway
        graph_feat = self.feature_projection(self.graph_features)
        static_repr = self.node_embedding + graph_feat

        # Neighbourhood enrichment
        neighbor_repr = torch.matmul(self.static_norm_adj, static_repr)
        gate_nb = torch.sigmoid(self.graph_neighbor_gate)
        static_repr = static_repr + gate_nb * neighbor_repr

        static_logits = self.static_routing

        if self.use_input_context:
            # Dynamic pathway
            h = self.input_encoder(x)
            attn_scores = self.temporal_attn(h)
            attn_weights = torch.softmax(attn_scores, dim=1)
            node_input_repr = (h * attn_weights).sum(dim=1)

            neighbor_repr = torch.matmul(self.norm_adj, node_input_repr)
            spatial_repr = self.spatial_mixer(neighbor_repr)
            dynamic_repr = node_input_repr + spatial_repr
            dynamic_repr = self.dynamic_proj(dynamic_repr)

            # Adaptive gating
            static_expanded = static_repr.unsqueeze(0).expand(B, -1, -1)
            gate_input = torch.cat([static_expanded, dynamic_repr], dim=-1)
            gate_weight = self.gate(gate_input)

            fused_repr = gate_weight * static_expanded + (1 - gate_weight) * dynamic_repr

            dynamic_logits = self.dynamic_router(fused_repr)
            final_logits = static_logits.unsqueeze(0) + dynamic_logits
        else:
            final_logits = static_logits.unsqueeze(0).expand(B, -1, -1)

        # Routing weights
        if self.early_routing and self.routing_mode == 'hard':
            _, expert_indices = final_logits.topk(self.top_k, dim=-1)
            if return_weights:
                routing_weights = self._noisy_topk(final_logits, self.top_k, self.training)
                return expert_indices, routing_weights, static_logits, graph_feat
            return expert_indices
        else:
            routing_weights = self._noisy_topk(final_logits, self.top_k, self.training)
            if return_weights:
                return routing_weights, static_logits, graph_feat
            return routing_weights


# ── Alternative Routers (for ablation) ──

class DenseMLPRouter(nn.Module):
    """Standard dense MoE router without graph topology features (ablation baseline)."""
    ROUTER_TYPE = 'dense_mlp'

    def __init__(self, num_experts, node_num, embed_dim=64,
                 input_dim=None, use_input_context=True,
                 temperature=1.0, top_k=None, noise_scale=0.5, **kwargs):
        super().__init__()
        self.num_experts = num_experts
        self.node_num = node_num
        self.temperature = temperature
        self.top_k = max(1, min(top_k if top_k is not None else num_experts, num_experts))
        self.noise_scale = noise_scale
        self.use_input_context = use_input_context

        # Learned node embeddings (no graph info)
        self.node_embedding = nn.Parameter(torch.randn(node_num, embed_dim) * 0.02)

        self.router_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_experts),
        )

        if use_input_context and input_dim is not None:
            self.input_proj = nn.Linear(input_dim, embed_dim) if input_dim != embed_dim else None
            self.input_encoder = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim, num_experts),
            )
        else:
            self.input_proj = None
            self.input_encoder = None

    def forward(self, x, return_weights=False):
        B, T, N, D = x.shape
        static_logits = self.router_mlp(self.node_embedding)  # [N, E]

        if self.use_input_context and self.input_encoder is not None:
            x_pooled = x.mean(dim=1)  # [B, N, D]
            if self.input_proj is not None:
                x_pooled = self.input_proj(x_pooled)
            dynamic_logits = self.input_encoder(x_pooled + self.node_embedding.unsqueeze(0))
            final_logits = static_logits.unsqueeze(0) + dynamic_logits
        else:
            final_logits = static_logits.unsqueeze(0).expand(B, -1, -1)

        routing_weights = self._topk_softmax(final_logits)
        if return_weights:
            return routing_weights, static_logits, self.node_embedding
        return routing_weights

    def _topk_softmax(self, logits):
        if self.top_k >= self.num_experts:
            return F.softmax(logits / self.temperature, dim=-1)
        if self.training:
            logits = logits + torch.randn_like(logits) * self.noise_scale
        topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)
        masked = torch.full_like(logits, float('-inf'))
        masked.scatter_(-1, topk_idx, topk_vals)
        return F.softmax(masked / self.temperature, dim=-1)


class SwitchRouter(nn.Module):
    """Switch Transformer style router (hard top-1 + STE). Fedus et al. (2022)."""
    ROUTER_TYPE = 'switch'

    def __init__(self, num_experts, node_num, embed_dim=64,
                 input_dim=None, use_input_context=True,
                 temperature=1.0, capacity_factor=1.25, **kwargs):
        super().__init__()
        self.num_experts = num_experts
        self.node_num = node_num
        self.temperature = temperature
        self.capacity_factor = capacity_factor

        self.node_embedding = nn.Parameter(torch.randn(node_num, embed_dim) * 0.02)
        self.router_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_experts),
        )

        if use_input_context and input_dim is not None:
            self.input_proj = nn.Linear(input_dim, embed_dim) if input_dim != embed_dim else None
            self.input_encoder = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, num_experts),
            )
        else:
            self.input_proj = None
            self.input_encoder = None

    def forward(self, x, return_weights=False):
        B, T, N, D = x.shape
        logits = self.router_mlp(self.node_embedding)  # [N, E]

        if self.input_encoder is not None:
            x_pooled = x.mean(dim=1)
            if self.input_proj is not None:
                x_pooled = self.input_proj(x_pooled)
            dynamic = self.input_encoder(x_pooled + self.node_embedding.unsqueeze(0))
            final_logits = logits.unsqueeze(0) + dynamic  # [B, N, E]
        else:
            final_logits = logits.unsqueeze(0).expand(B, -1, -1)

        # Hard top-1 with straight-through estimator
        soft_weights = F.softmax(final_logits / self.temperature, dim=-1)
        top1_idx = final_logits.argmax(dim=-1, keepdim=True)  # [B, N, 1]
        hard_weights = torch.zeros_like(soft_weights).scatter_(-1, top1_idx, 1.0)
        # STE: forward uses hard, backward uses soft
        routing_weights = hard_weights - soft_weights.detach() + soft_weights

        if return_weights:
            return routing_weights, logits, self.node_embedding
        return routing_weights


class ExpertChoiceRouter(nn.Module):
    """Expert Choice routing: each expert selects its tokens. Zhou et al. (NeurIPS 2022)."""
    ROUTER_TYPE = 'expert_choice'

    def __init__(self, num_experts, node_num, embed_dim=64,
                 input_dim=None, use_input_context=True,
                 temperature=1.0, top_k_ratio=0.5, **kwargs):
        super().__init__()
        self.num_experts = num_experts
        self.node_num = node_num
        self.temperature = temperature
        self.top_k = max(1, int(node_num * top_k_ratio))

        self.node_embedding = nn.Parameter(torch.randn(node_num, embed_dim) * 0.02)

        # Each expert has its own scorer
        self.expert_scorers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim // 2, 1),
            )
            for _ in range(num_experts)
        ])

        if use_input_context and input_dim is not None:
            self.input_proj = nn.Linear(input_dim, embed_dim) if input_dim != embed_dim else None
        else:
            self.input_proj = None

    def forward(self, x, return_weights=False):
        B, T, N, D = x.shape

        if self.input_proj is not None:
            x_pooled = x.mean(dim=1)  # [B, N, D]
            x_proj = self.input_proj(x_pooled)  # [B, N, embed]
            node_feats = x_proj + self.node_embedding.unsqueeze(0)  # [B, N, embed]
        else:
            node_feats = self.node_embedding.unsqueeze(0).expand(B, -1, -1)

        weights = torch.zeros(B, N, self.num_experts, device=x.device)

        for e, scorer in enumerate(self.expert_scorers):
            scores = scorer(node_feats).squeeze(-1)  # [B, N]

            # Expert picks top-k nodes
            topk_vals, topk_idx = torch.topk(scores, self.top_k, dim=-1)  # [B, top_k]
            soft_scores = F.softmax(topk_vals / self.temperature, dim=-1)

            # Scatter scores into mask
            mask = torch.zeros(B, N, device=x.device)
            mask.scatter_(1, topk_idx, soft_scores)
            weights[:, :, e] = mask

        # Normalize per node so weights sum to 1
        weight_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        weights = weights / weight_sum

        if return_weights:
            static_logits = torch.stack(
                [scorer(self.node_embedding.unsqueeze(0)).squeeze(-1).squeeze(0)
                 for scorer in self.expert_scorers], dim=-1
            )  # [N, E]
            return weights, static_logits, self.node_embedding
        return weights


class HashRouter(nn.Module):
    """Deterministic hash-based routing: node_id % num_experts. No learned params."""
    ROUTER_TYPE = 'hash'

    def __init__(self, num_experts, node_num, **kwargs):
        super().__init__()
        self.num_experts = num_experts
        self.node_num = node_num
        self.temperature = 1.0  # compatibility

        assignment = torch.zeros(node_num, num_experts)
        for i in range(node_num):
            assignment[i, i % num_experts] = 1.0
        self.register_buffer('assignment', assignment)
        # Dummy param for router param counting and optimizer compatibility
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x, return_weights=False):
        B = x.shape[0]
        weights = self.assignment.unsqueeze(0).expand(B, -1, -1)  # [B, N, E]
        if return_weights:
            return weights, self.assignment, self.assignment[:, :1]
        return weights


# ── Graph-Conditioned Output Refinement ──

class GraphConditionedOutputRefinement(nn.Module):
    """Per-node affine correction of MoE output, conditioned on graph topology.

    Maps topology features -> per-node (scale, bias) via hypernetworks,
    smooths through one graph convolution pass, and applies via gated residual.
    Zero-init + startup gate ensure expert predictions are preserved early in training.
    """

    def __init__(self, node_num, output_dim, adj_mx, hidden_dim=32, dropout=0.1,
                 routing_dim=0):
        super().__init__()
        self.node_num    = node_num
        self.output_dim  = output_dim
        self.routing_dim = routing_dim

        # Topology features from the shared module-level cache (0 extra compute)
        graph_feats = _compute_graph_features_cached(adj_mx, node_num)
        self.register_buffer('graph_feats', graph_feats)
        graph_feat_dim = graph_feats.shape[1]
        input_dim = graph_feat_dim + routing_dim

        # Normalized adjacency for spatial smoothing
        norm_adj = GraphConditionedExpertRouter._build_norm_adj(adj_mx, node_num)
        self.register_buffer('norm_adj', norm_adj)

        # Hypernetwork: topology+routing -> per-node scale
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Hypernetwork: topology+routing -> per-node bias
        self.bias_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Spatial mixer: graph convolution refinement
        self.spatial_refine = nn.Linear(output_dim, output_dim, bias=False)

        # Startup gate: sigmoid(-5) ~ 0.007, opens gradually
        self.gate = nn.Parameter(torch.tensor(-5.0))

        # Zero-init output layers -> correction = 0 at init
        nn.init.zeros_(self.scale_net[-1].weight)
        nn.init.zeros_(self.scale_net[-1].bias)
        nn.init.zeros_(self.bias_net[-1].weight)
        nn.init.zeros_(self.bias_net[-1].bias)
        if output_dim == self.spatial_refine.weight.shape[-1]:
            nn.init.eye_(self.spatial_refine.weight)
        else:
            nn.init.xavier_uniform_(self.spatial_refine.weight, gain=0.01)

    # ---------------------------------------------------------------------- #
    def forward(self, x, routing_weights=None):
        B, T, N, D = x.shape

        # Build conditioning: graph features + optional routing weights
        if routing_weights is not None and self.routing_dim > 0:
            rw_avg = routing_weights.detach().mean(dim=0)
            cond = torch.cat([self.graph_feats, rw_avg], dim=-1)
        else:
            cond = self.graph_feats

        # Per-node affine correction, bounded via tanh (at most +/- 30%)
        scale = 0.3 * torch.tanh(self.scale_net(cond))
        bias  = 0.3 * torch.tanh(self.bias_net(cond))

        correction = (x * scale.unsqueeze(0).unsqueeze(0)
                      + bias.unsqueeze(0).unsqueeze(0))

        # Spatial smoothing via graph convolution
        corr_flat = correction.reshape(B * T, N, D)
        spatial = torch.matmul(self.norm_adj, corr_flat)
        spatial = self.spatial_refine(spatial)
        spatial = spatial.reshape(B, T, N, D)

        # Gated residual
        return x + torch.sigmoid(self.gate) * spatial


# ── Router Registry ──
ROUTER_REGISTRY = {
    'graph_conditioned': GraphConditionedExpertRouter,
    'dense_mlp': DenseMLPRouter,
    'switch': SwitchRouter,
    'expert_choice': ExpertChoiceRouter,
    'hash': HashRouter,
}


def create_router(router_type, num_experts, node_num, graph_features=None,
                  embed_dim=32, use_input_context=True, temperature=1.0,
                  input_dim=None, top_k=None, noise_scale=0.5,
                  early_routing=False, routing_mode='soft', adj_mx=None):
    """Factory function to create a router by type name."""
    if router_type not in ROUTER_REGISTRY:
        raise ValueError(
            f"Unknown router_type '{router_type}'. "
            f"Available: {list(ROUTER_REGISTRY.keys())}"
        )

    cls = ROUTER_REGISTRY[router_type]

    if router_type == 'graph_conditioned':
        return cls(
            num_experts=num_experts,
            node_num=node_num,
            graph_features=graph_features,
            embed_dim=embed_dim,
            use_input_context=use_input_context,
            temperature=temperature,
            input_dim=input_dim,
            top_k=top_k,
            noise_scale=noise_scale,
            early_routing=early_routing,
            routing_mode=routing_mode,
            adj_mx=adj_mx,
        )
    elif router_type == 'hash':
        return cls(num_experts=num_experts, node_num=node_num)
    else:
        return cls(
            num_experts=num_experts,
            node_num=node_num,
            embed_dim=embed_dim,
            input_dim=input_dim,
            use_input_context=use_input_context,
            temperature=temperature,
            top_k=top_k,
            noise_scale=noise_scale,
        )


class MoE_STLoRA(nn.Module):
    """MoE with Graph-Conditioned Routing and optional ST-LoRA adapters."""
    
    def __init__(self, device, node_num, input_dim, output_dim, horizon,
                 expert_models, expert_names, supports, adj_mx,
                 frozen_experts=True, shared_adapters=False,
                 embed_dim=12, num_layers=4, num_blocks=1,
                 la_dropout=0.3, last_lr=1e-4, last_weight_decay=1e-5,
                 router_embed_dim=32, use_input_context=True, temperature=1.0,
                 load_balance_weight=0.01, top_k=None, noise_scale=0.5,
                 lagcn=False, linear=False, early_routing=False, routing_mode='soft',
                 input_projection=False, shared_input_proj=False, input_proj_hidden=32,
                 input_proj_dropout=0.1, router_type='graph_conditioned',
                 output_refinement=False, refine_hidden_dim=32,
                 routing_entropy_weight=0.5):
        super().__init__()
        self.device = device
        self.num_node = node_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_experts = len(expert_models)
        self.expert_names = expert_names or [f"Expert_{i}" for i in range(self.num_experts)]
        self.frozen_experts = frozen_experts
        self.shared_adapters = shared_adapters
        self.supports = supports
        self.lagcn = lagcn
        self.linear = linear
        self.load_balance_weight = load_balance_weight
        self.routing_entropy_weight = routing_entropy_weight
        self.early_routing = early_routing
        self.routing_mode = routing_mode
        self.use_input_projection = input_projection
        self.router_type = router_type
        
        self.graph_features = GraphTopologyFeatures(adj_mx, device)
        
        self.experts = nn.ModuleList(expert_models)
        
        if frozen_experts:
            for expert in self.experts:
                for param in expert.parameters():
                    param.requires_grad = False
        
        if shared_adapters:
            self.adapters = self._create_adapter_blocks(
                num_blocks, embed_dim, num_layers, la_dropout, linear
            )
        else:
            self.adapters = nn.ModuleList([
                self._create_adapter_blocks(
                    num_blocks, embed_dim, num_layers, la_dropout, linear
                )
                for _ in range(self.num_experts)
            ])
        
        # Adapter gate: sigmoid(-3) ~ 0.05, opens during training
        num_adapter_sets = 1 if shared_adapters else self.num_experts
        self.adapter_gate = nn.Parameter(torch.full((num_adapter_sets,), -3.0))
        
        self._init_adapters_small()
        
        self.router = create_router(
            router_type=router_type,
            num_experts=self.num_experts,
            node_num=node_num,
            graph_features=self.graph_features,
            embed_dim=router_embed_dim,
            use_input_context=use_input_context,
            temperature=temperature,
            input_dim=input_dim,
            top_k=top_k,
            noise_scale=noise_scale,
            early_routing=early_routing,
            routing_mode=routing_mode,
            adj_mx=adj_mx,
        )
        
        # Input projection (optional)
        self.input_proj = None
        if input_projection:
            self.input_proj = NodeInputProjection(
                num_experts=self.num_experts,
                node_num=node_num,
                input_dim=input_dim,
                adj_mx=adj_mx,
                proj_hidden_dim=input_proj_hidden,
                shared_projection=shared_input_proj,
                dropout=input_proj_dropout,
            )
        
        if lagcn:
            if supports is None or len(supports) == 0:
                raise ValueError("lagcn=True requires non-empty `supports`.")
            
            self.gconv = GCN(c_in=input_dim, c_out=output_dim, 
                           dropout=0.1, support_len=len(supports))

        # Output refinement (optional)
        self.output_refine = None
        if output_refinement:
            self.output_refine = GraphConditionedOutputRefinement(
                node_num=node_num,
                output_dim=output_dim,
                adj_mx=adj_mx,
                hidden_dim=refine_hidden_dim,
                dropout=0.1,
                routing_dim=self.num_experts,
            )
        
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=last_lr, 
                                         weight_decay=last_weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                         step_size=10, gamma=0.95)
    
    def _init_adapters_small(self):
        """Initialize adapter weights small so corrections start near-zero."""
        adapter_list = [self.adapters] if self.shared_adapters else self.adapters
        for adapter_block in adapter_list:
            for name, p in adapter_block.named_parameters():
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p, gain=0.01)
                elif p.dim() == 1:
                    nn.init.zeros_(p)
    
    def _create_adapter_blocks(self, num_blocks, embed_dim, num_layers, dropout, linear):
        """Create adapter blocks (ST-LoRA style)"""
        blocks = nn.ModuleList()
        for _ in range(num_blocks):
            blocks.append(
                Node_Specific_Predictor(
                    input_dim=self.output_dim,
                    output_dim=self.output_dim,
                    horizon=self.horizon,
                    hidden_dim=self.output_dim * embed_dim,
                    num_layers=num_layers,
                    supports=self.supports,
                    lora_dropout=dropout,
                    linear=linear,
                    lagcn=False
                )
            )
        return blocks
    
    def _apply_adapters(self, expert_output, adapter_blocks, expert_idx=0):
        """Apply ST-LoRA adaptation with gated residual."""
        original_output = expert_output
        
        tunings = []
        for adapter in adapter_blocks:
            adapter_out = adapter(expert_output)
            
            # Align temporal dim if needed
            if adapter_out.shape != original_output.shape:
                min_T = min(adapter_out.shape[1], original_output.shape[1])
                adapter_out = adapter_out[:, -min_T:, :, :]
                ref = original_output[:, -min_T:, :, :]
            else:
                ref = original_output
            
            # Adapter correction = adapter_out - ref (what the adapter wants to change)
            tunings.append(adapter_out - ref)
        
        if len(tunings) > 0:
            tuning_stack = torch.stack(tunings, dim=0)
            aggregated_correction = torch.mean(tuning_stack, dim=0)
            
            # Learnable gate controls how much correction to apply
            gate_idx = 0 if self.shared_adapters else expert_idx
            gate = torch.sigmoid(self.adapter_gate[gate_idx])
            
            if aggregated_correction.shape == original_output.shape:
                adapted_output = original_output + gate * aggregated_correction
            else:
                min_T = aggregated_correction.shape[1]
                adapted_output = original_output.clone()
                adapted_output[:, -min_T:, :, :] = (
                    adapted_output[:, -min_T:, :, :] + gate * aggregated_correction
                )
        else:
            adapted_output = original_output
            
        return adapted_output
    
    def _call_expert(self, expert, x, label=None, iter=None):
        """Call an expert with flexible signature; uses no_grad when frozen."""
        def _inner():
            try:
                return expert(x, label, iter)
            except TypeError:
                try:
                    return expert(x, label)
                except TypeError:
                    try:
                        return expert(x, iter)
                    except TypeError:
                        return expert(x)
        if self.frozen_experts:
            with torch.no_grad():
                return _inner()
        return _inner()
    
    def forward(self, x, label=None, iter=None, return_analysis=False):
        B, T, N, D = x.shape
        
        # Per-expert input projections (if enabled)
        if self.input_proj is not None:
            expert_inputs = self.input_proj(x)
        else:
            expert_inputs = [x] * self.num_experts
        
        if self.early_routing and self.routing_mode == 'hard':
            router_out = self.router(x, return_weights=True)
            # GraphConditionedExpertRouter returns 4 values;
            # other routers return 3 with return_weights=True.
            expert_indices, routing_weights = router_out[0], router_out[1]
            
            unique_experts = torch.unique(expert_indices)
            expert_outputs = {}
            
            for expert_idx in unique_experts:
                expert_idx = expert_idx.item()
                expert = self.experts[expert_idx]
                expert_x = expert_inputs[expert_idx]  # per-expert projected input
                
                output = self._call_expert(expert, expert_x, label, iter)
                
                if self.shared_adapters:
                    adapted = self._apply_adapters(output, self.adapters, expert_idx=0)
                else:
                    adapted = self._apply_adapters(output, self.adapters[expert_idx], expert_idx=expert_idx)
                
                if adapted.shape[1] > self.horizon:
                    adapted = adapted[:, -self.horizon:, :, :]
                elif adapted.shape[1] < self.horizon:
                    pad_size = self.horizon - adapted.shape[1]
                    padding = torch.zeros(adapted.shape[0], pad_size, adapted.shape[2], adapted.shape[3], 
                                        device=adapted.device, dtype=adapted.dtype)
                    adapted = torch.cat([padding, adapted], dim=1)
                
                expert_outputs[expert_idx] = adapted
            
            batch_outputs = []
            for b in range(B):
                node_outputs = []
                for n in range(N):
                    selected_indices = expert_indices[b, n]
                    node_output = torch.zeros(self.horizon, 1, self.output_dim, device=x.device)
                    weights_sum = 0.0
                    
                    for k_idx in range(selected_indices.shape[0]):
                        expert_idx = selected_indices[k_idx].item()
                        weight = routing_weights[b, n, expert_idx]
                        node_output += weight * expert_outputs[expert_idx][b, :, n, :]
                        weights_sum += weight
                    
                    if weights_sum > 0:
                        node_output = node_output / weights_sum
                    node_outputs.append(node_output)
                
                batch_outputs.append(torch.stack(node_outputs, dim=1))
            
            final_output = torch.stack(batch_outputs, dim=0)
            
            all_expert_outputs = [expert_outputs.get(i, torch.zeros_like(list(expert_outputs.values())[0])) 
                                  for i in range(self.num_experts)]
        else:
            routing_weights = self.router(x)
            
            expert_outputs = []
            for i, expert in enumerate(self.experts):
                expert_x = expert_inputs[i]  # per-expert projected input
                output = self._call_expert(expert, expert_x, label, iter)
                
                if self.shared_adapters:
                    adapted = self._apply_adapters(output, self.adapters, expert_idx=0)
                else:
                    adapted = self._apply_adapters(output, self.adapters[i], expert_idx=i)
                
                if adapted.shape[1] > self.horizon:
                    adapted = adapted[:, -self.horizon:, :, :]
                elif adapted.shape[1] < self.horizon:
                    pad_size = self.horizon - adapted.shape[1]
                    padding = torch.zeros(adapted.shape[0], pad_size, adapted.shape[2], adapted.shape[3], 
                                        device=adapted.device, dtype=adapted.dtype)
                    adapted = torch.cat([padding, adapted], dim=1)
                
                expert_outputs.append(adapted)
            
            # stacked: [E, B, T, N, D] -> [B, N, E, T, D]
            stacked_outputs = torch.stack(expert_outputs, dim=0)
            stacked_outputs = stacked_outputs.permute(1, 3, 0, 2, 4)

            # Weighted sum over experts
            routing_weights_expanded = routing_weights.unsqueeze(-1).unsqueeze(-1)
            weighted_outputs = (stacked_outputs * routing_weights_expanded).sum(dim=2)
            final_output = weighted_outputs.transpose(1, 2)
            all_expert_outputs = expert_outputs

        # ── Graph-conditioned output refinement (if enabled) ──────────────────
        # Applied to the final weighted combination, BEFORE any lagcn pass.
        # Routing weights are passed so the module can condition on which expert
        # dominated each node — gives richer signal than graph topology alone.
        if self.output_refine is not None:
            final_output = self.output_refine(final_output, routing_weights=routing_weights)

        if self.lagcn:
            gcn_out = self.gconv(x, self.supports)[:, -self.horizon:, :, -self.output_dim:]
            final_output = final_output + gcn_out
        
        if return_analysis:
            return final_output, all_expert_outputs, routing_weights
        return final_output
    
    def compute_load_balance_loss(self, routing_weights, add_importance_loss=True, add_entropy_loss=False,
                                   entropy_weight=0.5):
        """Load balancing loss (Switch Transformer formulation, Fedus et al. 2022)."""
        E = self.num_experts
        B, N, _ = routing_weights.shape

        # f_e: hard dispatch fraction per expert
        expert_indices = routing_weights.argmax(dim=-1)
        one_hot = F.one_hot(expert_indices, num_classes=E).float()
        f_e = one_hot.mean(dim=[0, 1])

        # P_e: mean routing probability per expert
        P_e = routing_weights.mean(dim=[0, 1])

        balance_loss = E * (f_e * P_e).sum()

        total_loss = balance_loss

        # CV^2 penalty on expert importance
        if add_importance_loss:
            importance_mean = P_e.mean()
            importance_std = P_e.std()
            cv2 = (importance_std / (importance_mean + 1e-8)) ** 2
            total_loss = total_loss + 0.1 * cv2

        # Entropy minimisation: push each node toward sharp expert preference
        if add_entropy_loss:
            entropy = -(routing_weights * torch.log(routing_weights + 1e-9)).sum(dim=-1)
            total_loss = total_loss + entropy_weight * entropy.mean()

        return total_loss
    
    def get_expert_statistics(self, return_per_node=False):
        """Analyze expert routing patterns."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.num_node, self.input_dim, device=self.device)
            _, static_routing, node_embeds = self.router(
                dummy_input,
                return_weights=True
            )
            
            expert_weights = F.softmax(static_routing / self.router.temperature, dim=-1)
            avg_weights = expert_weights.mean(dim=0)
            
            preferred_expert = expert_weights.argmax(dim=1)
            
            stats = {
                'expert_avg_weights': {
                    name: weight.item() 
                    for name, weight in zip(self.expert_names, avg_weights)
                },
                'expert_max_nodes': {
                    name: (preferred_expert == i).sum().item()
                    for i, name in enumerate(self.expert_names)
                },
                'routing_entropy': -(expert_weights * torch.log(expert_weights + 1e-9)).sum(dim=1).mean().item()
            }
            
            if return_per_node:
                stats['per_node_routing'] = expert_weights.cpu().numpy()
                stats['node_embeddings'] = node_embeds.cpu().numpy()
                stats['preferred_expert'] = preferred_expert.cpu().numpy()
            
            return stats
    
    def visualize_routing(self):
        """Print routing statistics to console"""
        stats = self.get_expert_statistics()
        
        print("\n" + "="*70)
        print("Expert Routing Statistics (Graph-Based)")
        print("="*70)
        
        print("\nAverage Expert Weights:")
        for name, weight in stats['expert_avg_weights'].items():
            bar = "█" * int(weight * 50)
            print(f"  {name:20s}: {weight:.4f} {bar}")
        
        print(f"\nNodes Preferring Each Expert:")
        for name, count in stats['expert_max_nodes'].items():
            pct = 100 * count / self.num_node
            bar = "█" * int(pct / 2)
            print(f"  {name:20s}: {count:4d}/{self.num_node} nodes ({pct:5.1f}%) {bar}")
        
        print(f"\nRouting Entropy: {stats['routing_entropy']:.4f}")
        print(f"  (Higher = more diverse routing, Lower = more specialized)")
        print("="*70 + "\n")


def create_moe_stlora_from_config(args, expert_models, expert_names=None):
    return MoE_STLoRA(
        device=args.device,
        node_num=args.node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        horizon=args.horizon,
        expert_models=expert_models,
        expert_names=expert_names,
        supports=args.supports if hasattr(args, 'supports') else None,
        adj_mx=args.adj_mx,
        frozen_experts=getattr(args, 'frozen_experts', True),
        shared_adapters=getattr(args, 'shared_adapters', False),
        embed_dim=getattr(args, 'embed_dim', 12),
        num_layers=getattr(args, 'num_layers', 4),
        num_blocks=getattr(args, 'num_blocks', 1),
        la_dropout=getattr(args, 'last_dropout', 0.3),
        last_lr=getattr(args, 'last_lr', 1e-4),
        last_weight_decay=getattr(args, 'last_weight_decay', 1e-5),
        router_embed_dim=getattr(args, 'router_embed_dim', 32),
        use_input_context=getattr(args, 'use_input_context', True),
        temperature=getattr(args, 'moe_temperature', 1.0),
        load_balance_weight=getattr(args, 'load_balance_weight', 0.01),
        top_k=getattr(args, 'moe_top_k', None),
        noise_scale=getattr(args, 'moe_noise_scale', 0.5),
        lagcn=getattr(args, 'lagcn', False),
        linear=getattr(args, 'linear', False),
        early_routing=getattr(args, 'early_routing', False),
        routing_mode=getattr(args, 'routing_mode', 'soft'),
        input_projection=getattr(args, 'input_projection', False),
        shared_input_proj=getattr(args, 'shared_input_proj', False),
        input_proj_hidden=getattr(args, 'input_proj_hidden', 32),
        input_proj_dropout=getattr(args, 'input_proj_dropout', 0.1),
        router_type=getattr(args, 'router_type', 'graph_conditioned'),
        output_refinement=getattr(args, 'output_refinement', False),
        refine_hidden_dim=getattr(args, 'refine_hidden_dim', 32),
        routing_entropy_weight=getattr(args, 'routing_entropy_weight', 0.5),
    )
