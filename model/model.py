# model/model.py
# -*- coding: utf-8 -*-
"""
BindingAffinityModel
- ProteinBlockEncoder (EGNN-style)
- LigandBlockEncoder  (EGNN-style)
- CrossInteractionModule (bi-directional prot<->lig message passing)
- ComplexReadout (pool + priors -> MLP -> scalar)

This module:
- defines model only (no training loop, no logging in forward)
- returns pred shape (B,)
- designed for DDP wrapping and AMP autocast outside.

Author: yk + gpt
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# 0. Config dataclass
# ---------------------------------------------------------
@dataclass
class ModelConfig:
    hidden_dim: int = 32          # node hidden dim after input proj
    edge_dim: int = 16            # edge_attr dim (RBF encoded dist)
    num_layers_prot: int = 4      # EGNN layers for protein blocks
    num_layers_lig: int = 4       # EGNN layers for ligand blocks
    num_layers_cross: int = 2     # cross interaction steps
    dropout: float = 0.1
    use_priors: bool = True       # include priors (pose/confidence etc.) in readout
    use_gradient_checkpointing: bool = True  # stub, not wired yet
    use_pose_mil: bool = False    # MIL over poses (stub, single-pose default)
    use_assay_cond: bool = False  # condition on assay type (stub)
    # flexible input dims so we don't break if upstream changes
    prot_in_dim: int = 23         # prot_block_x dim
    lig_in_dim: int = 771         # lig_block_x dim (can be 387 or 771; we just project to hidden_dim)
    priors_dim: int = 1           # priors feature dim per sample
    assay_dim: int = 1            # (optional) assay code dim per sample

    readout_hidden_dim: int = 64  # MLP head hidden
    readout_layers: int = 2       # # of hidden layers in final MLP


# ---------------------------------------------------------
# 1. Small helpers
# ---------------------------------------------------------
def _rbf_distance(x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise distance magnitude ||x_j - x_i|| for edges.
    x_i, x_j: (E, 3)
    returns: (E, 1)
    """
    diff = x_j - x_i
    dist = torch.sqrt(torch.sum(diff * diff, dim=-1, keepdim=True) + 1e-9)
    return dist  # (E,1)


class MLP(nn.Module):
    """Simple MLP with configurable depth."""
    def __init__(self, in_dim, hidden_dim, out_dim, n_hidden_layers=1, dropout=0.0, act=nn.SiLU):
        super().__init__()
        layers = []
        dim_prev = in_dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(dim_prev, hidden_dim))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim_prev = hidden_dim
        layers.append(nn.Linear(dim_prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------
# 2. EGNN-style layer (per-molecule)
# ---------------------------------------------------------
class EGNNLayer(nn.Module):
    """
    Minimal EGNN-ish block:
    Inputs:
        h: (N, H)
        x: (N, 3)
        edge_index: (2, E)
        edge_attr: (E, edge_dim)

    Steps:
        - for each edge (i->j):
            m_ij = phi_e([h_i, h_j, edge_attr_ij, ||x_j-x_i||])
        - agg_j = sum_i m_ij (incoming messages)
        - h_j' = phi_h([h_j, agg_j])
        - x_j' = x_j + sum_i phi_x(m_ij) * (x_j - x_i)  (directional update)
          (we scale by normalized direction to encourage equivariance-ish behavior)
    """
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.dropout = dropout

        self.phi_e = MLP(
            in_dim=hidden_dim * 2 + edge_dim + 1,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            n_hidden_layers=1,
            dropout=dropout,
            act=nn.SiLU
        )

        self.phi_h = MLP(
            in_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            n_hidden_layers=1,
            dropout=dropout,
            act=nn.SiLU
        )

        # coordinate update uses a scalar gate per message
        self.phi_x = MLP(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=1,
            n_hidden_layers=1,
            dropout=dropout,
            act=nn.SiLU
        )

        self.norm_h = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h: (N,H)
        x: (N,3)
        edge_index: (2,E) long
        edge_attr: (E,edge_dim)
        """
        src = edge_index[0]  # (E,)
        dst = edge_index[1]  # (E,)

        h_i = h[src]  # (E,H)
        h_j = h[dst]  # (E,H)

        x_i = x[src]  # (E,3)
        x_j = x[dst]  # (E,3)

        dist_ij = _rbf_distance(x_i, x_j)  # (E,1)

        #m_ij_input = torch.cat([h_i, h_j, edge_attr, dist_ij], dim=-1)  # (E, 2H+edge_dim+1)
        # unify dtypes before cat
        if edge_attr.dtype != h.dtype:
            edge_attr = edge_attr.to(h.dtype)
        if dist_ij.dtype != h.dtype:
            dist_ij = dist_ij.to(h.dtype)
        m_ij_input = torch.cat([h_i, h_j, edge_attr, dist_ij], dim=-1)
        m_ij = self.phi_e(m_ij_input)  # (E,H)

        # aggregate messages per dst node
        N = h.shape[0]
        agg = torch.zeros(N, self.hidden_dim, device=h.device, dtype=h.dtype)
        # ensure dtype match under AMP (bf16/fp16)
        if m_ij.dtype != h.dtype:
            m_ij = m_ij.to(h.dtype)
        agg.index_add_(0, dst, m_ij)

        # feature update
        h_cat = torch.cat([h, agg], dim=-1)  # (N,2H)
        dh = self.phi_h(h_cat)
        if dh.dtype != h.dtype:
            dh = dh.to(h.dtype)
        h_out = self.norm_h(h + dh)

        # coordinate update
        # directional vector (x_j - x_i)
        direction = (x_j - x_i)  # (E,3)
        # normalize direction to avoid exploding updates
        direction_norm = torch.norm(direction, dim=-1, keepdim=True) + 1e-9
        direction_unit = direction / direction_norm  # (E,3)

        gate_ij = self.phi_x(m_ij)  # (E,1)
        coord_msg = direction_unit * gate_ij  # (E,3)

        dx = torch.zeros(N, 3, device=x.device, dtype=x.dtype)
        if coord_msg.dtype != dx.dtype:
            coord_msg = coord_msg.to(dx.dtype)
        dx.index_add_(0, dst, coord_msg)

        if dx.dtype != x.dtype:
            dx = dx.to(x.dtype)
        x_out = x + dx  # (N,3)

        return h_out, x_out


# ---------------------------------------------------------
# 3. Cross Interaction Module
# ---------------------------------------------------------
class CrossInteractionModule(nn.Module):
    """
    Bi-directional protein <-> ligand message passing.
    We'll build temporary cross edges, but we assume caller gives:
        cross_index: (2, Ec)
            index 0 = prot node idx
            index 1 = lig node idx
        cross_attr: (Ec, edge_dim)

    We'll update both prot_h and lig_h using messages from each other.
    We DO NOT update coordinates here for now (simplify);
    this keeps prot_x / lig_x stable post-encoders.
    """
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            _CrossLayer(hidden_dim, edge_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        prot_h: torch.Tensor,
        lig_h: torch.Tensor,
        cross_index: torch.Tensor,
        cross_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            prot_h, lig_h = layer(prot_h, lig_h, cross_index, cross_attr)
        return prot_h, lig_h


class _CrossLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim

        # prot <- lig
        self.msg_pl = MLP(
            in_dim=hidden_dim * 2 + edge_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            n_hidden_layers=1,
            dropout=dropout,
            act=nn.SiLU
        )
        self.upd_p = MLP(
            in_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            n_hidden_layers=1,
            dropout=dropout,
            act=nn.SiLU
        )
        self.norm_p = nn.LayerNorm(hidden_dim)

        # lig <- prot
        self.msg_lp = MLP(
            in_dim=hidden_dim * 2 + edge_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            n_hidden_layers=1,
            dropout=dropout,
            act=nn.SiLU
        )
        self.upd_l = MLP(
            in_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            n_hidden_layers=1,
            dropout=dropout,
            act=nn.SiLU
        )
        self.norm_l = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        prot_h: torch.Tensor,   # (Np,H)
        lig_h: torch.Tensor,    # (Nl,H)
        cross_index: torch.Tensor,  # (2,Ec): [prot_idx, lig_idx]
        cross_attr: torch.Tensor    # (Ec, edge_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prot_idx = cross_index[0]  # (Ec,)
        lig_idx = cross_index[1]  # (Ec,)

        p_feat = prot_h[prot_idx]  # (Ec,H)
        l_feat = lig_h[lig_idx]    # (Ec,H)

        # unify dtypes before cat (AMP compatibility)
        if cross_attr.dtype != prot_h.dtype:
            cross_attr = cross_attr.to(prot_h.dtype)

        # messages to protein from ligand
        pl_in = torch.cat([p_feat, l_feat, cross_attr], dim=-1)  # (Ec,2H+edge_dim)
        m_pl = self.msg_pl(pl_in)  # (Ec,H)

        # messages to ligand from protein
        lp_in = torch.cat([l_feat, p_feat, cross_attr], dim=-1)  # symmetric concat
        m_lp = self.msg_lp(lp_in)  # (Ec,H)

        # aggregate per prot node
        Np = prot_h.shape[0]
        agg_p = torch.zeros(Np, self.hidden_dim, device=prot_h.device, dtype=prot_h.dtype)
        if m_pl.dtype != prot_h.dtype:
            m_pl = m_pl.to(prot_h.dtype)
        agg_p.index_add_(0, prot_idx, m_pl)

        # aggregate per lig node
        Nl = lig_h.shape[0]
        agg_l = torch.zeros(Nl, self.hidden_dim, device=lig_h.device, dtype=lig_h.dtype)
        if m_lp.dtype != lig_h.dtype:
            m_lp = m_lp.to(lig_h.dtype)
        agg_l.index_add_(0, lig_idx, m_lp)

        # update
        p_up_in = torch.cat([prot_h, agg_p], dim=-1)
        l_up_in = torch.cat([lig_h, agg_l], dim=-1)

        dp = self.upd_p(p_up_in)
        dl = self.upd_l(l_up_in)

        # ensure dtype match for residual add
        if dp.dtype != prot_h.dtype:
            dp = dp.to(prot_h.dtype)
        if dl.dtype != lig_h.dtype:
            dl = dl.to(lig_h.dtype)

        prot_h_out = self.norm_p(prot_h + dp)
        lig_h_out = self.norm_l(lig_h + dl)

        return prot_h_out, lig_h_out


# ---------------------------------------------------------
# 4. Pooling + Readout
# ---------------------------------------------------------
class ComplexReadout(nn.Module):
    """
    Takes per-node embeddings + batch vectors -> pooled complex embedding.
    Optionally concatenates priors (pose confidence etc.).
    Then regression MLP -> (B,)
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # final feature dim before head
        base_dim = cfg.hidden_dim * 2  # [prot_pool || lig_pool]

        if cfg.use_priors:
            base_dim += cfg.priors_dim

        if cfg.use_assay_cond:
            base_dim += cfg.assay_dim

        # MLP head
        layers = []
        in_dim = base_dim
        for _ in range(cfg.readout_layers):
            layers.append(nn.Linear(in_dim, cfg.readout_hidden_dim))
            layers.append(nn.SiLU())
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = cfg.readout_hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        prot_h: torch.Tensor,
        lig_h: torch.Tensor,
        prot_batch: torch.Tensor,
        lig_batch: torch.Tensor,
        priors: Optional[torch.Tensor] = None,
        assay_code: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        prot_h: (Np,H)
        lig_h:  (Nl,H)
        prot_batch: (Np,) long -> which complex each protein node belongs to
        lig_batch:  (Nl,) long -> which complex each ligand node belongs to
        priors: (B,P) or None
        assay_code: (B,A) or None

        returns pred: (B,)
        """

        B = max(int(prot_batch.max().item()), int(lig_batch.max().item())) + 1

        # mean-pool protein / ligand features per complex
        device = prot_h.device
        dtype = prot_h.dtype

        prot_sum = torch.zeros(B, prot_h.size(-1), device=device, dtype=dtype)
        prot_cnt = torch.zeros(B, 1, device=device, dtype=dtype)
        # ensure prot_h matches dtype for index_add_
        if prot_h.dtype != dtype:
            prot_h = prot_h.to(dtype)
        prot_sum.index_add_(0, prot_batch, prot_h)
        prot_cnt.index_add_(0, prot_batch, torch.ones_like(prot_batch, dtype=dtype).unsqueeze(-1))
        prot_pool = prot_sum / (prot_cnt.clamp(min=1.0))  # (B,H)

        lig_sum = torch.zeros(B, lig_h.size(-1), device=device, dtype=dtype)
        lig_cnt = torch.zeros(B, 1, device=device, dtype=dtype)
        # ensure lig_h matches dtype for index_add_
        if lig_h.dtype != dtype:
            lig_h = lig_h.to(dtype)
        lig_sum.index_add_(0, lig_batch, lig_h)
        lig_cnt.index_add_(0, lig_batch, torch.ones_like(lig_batch, dtype=dtype).unsqueeze(-1))
        lig_pool = lig_sum / (lig_cnt.clamp(min=1.0))      # (B,H)

        feats = [prot_pool, lig_pool]

        if self.cfg.use_priors and priors is not None:
            # priors expected (B,P). We'll just concat.
            feats.append(priors.to(device=device, dtype=dtype))

        if self.cfg.use_assay_cond and assay_code is not None:
            feats.append(assay_code.to(device=device, dtype=dtype))

        complex_feat = torch.cat(feats, dim=-1)  # (B, dim)

        pred = self.mlp(complex_feat).squeeze(-1)  # (B,)
        return pred


# ---------------------------------------------------------
# 5. Encoder modules
# ---------------------------------------------------------
class BlockEncoder(nn.Module):
    """
    Generic block encoder for either protein or ligand.
    1) Project input node features -> hidden_dim
    2) Repeated EGNN layers with coord updates
    """
    def __init__(self, in_dim: int, cfg: ModelConfig, num_layers: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, cfg.hidden_dim)
        self.layers = nn.ModuleList([
            EGNNLayer(cfg.hidden_dim, cfg.edge_dim, cfg.dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        node_pos: torch.Tensor,        # (N,3)
        node_feat: torch.Tensor,       # (N,in_dim)
        edge_index: torch.Tensor,      # (2,E)
        edge_attr: torch.Tensor        # (E,edge_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.proj(node_feat)  # (N,H)
        x = node_pos              # (N,3)
        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr)
        return h, x


# ---------------------------------------------------------
# 6. Full Model
# ---------------------------------------------------------
class BindingAffinityModel(nn.Module):
    """
    High-level wrapper.
    forward(...) -> pred (B,)

    MIL / multi-pose:
    - Right now: assume batch already corresponds to poses we actually train on.
    - If cfg.use_pose_mil == True, we would pool multiple poses per entry_id
      using attention. Stubbed for future extension.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # encoders
        self.prot_encoder = BlockEncoder(
            in_dim=cfg.prot_in_dim,
            cfg=cfg,
            num_layers=cfg.num_layers_prot
        )
        self.lig_encoder = BlockEncoder(
            in_dim=cfg.lig_in_dim,
            cfg=cfg,
            num_layers=cfg.num_layers_lig
        )

        # cross interaction
        self.cross_module = CrossInteractionModule(
            hidden_dim=cfg.hidden_dim,
            edge_dim=cfg.edge_dim,
            dropout=cfg.dropout,
            num_layers=cfg.num_layers_cross
        )

        # readout head
        self.readout = ComplexReadout(cfg)

        # stubs for MIL / assay conditioning
        # (attention pooling over poses, assay embedding, etc.)
        # For now just identity.

    def forward(
        self,
        prot_pos: torch.Tensor,             # (Np,3)
        prot_x: torch.Tensor,               # (Np,prot_in_dim)
        prot_edge_index: torch.Tensor,      # (2,Ep)
        prot_edge_attr: torch.Tensor,       # (Ep,edge_dim)
        prot_batch: torch.Tensor,           # (Np,)

        lig_pos: torch.Tensor,              # (Nl,3)
        lig_x: torch.Tensor,                # (Nl,lig_in_dim)
        lig_edge_index: torch.Tensor,       # (2,El)
        lig_edge_attr: torch.Tensor,        # (El,edge_dim)
        lig_batch: torch.Tensor,            # (Nl,)

        cross_index: torch.Tensor,          # (2,Ec) 0=prot_idx,1=lig_idx
        cross_attr: torch.Tensor,           # (Ec,edge_dim)

        priors: Optional[torch.Tensor] = None,   # (B,P)
        assay: Optional[torch.Tensor] = None     # (B,A) or codes
    ) -> torch.Tensor:
        """
        Returns:
            pred: (B,) float32
        """

        # Encode protein / ligand separately (EGNN over intra-edges)
        prot_h, prot_pos_upd = self.prot_encoder(
            prot_pos, prot_x, prot_edge_index, prot_edge_attr
        )
        lig_h, lig_pos_upd = self.lig_encoder(
            lig_pos, lig_x, lig_edge_index, lig_edge_attr
        )

        # Cross interaction (bi-directional message passing)
        prot_h_cross, lig_h_cross = self.cross_module(
            prot_h, lig_h, cross_index, cross_attr
        )

        # (Optional) MIL across poses would go here if cfg.use_pose_mil.
        # Right now: assume prot_batch / lig_batch already maps 1 pose -> 1 sample.
        # We just mean-pool per batch.
        assay_feat = None
        if self.cfg.use_assay_cond and assay is not None:
            # we'll trust caller to give already-embedded assay of shape (B,A)
            assay_feat = assay

        pred = self.readout(
            prot_h_cross,
            lig_h_cross,
            prot_batch,
            lig_batch,
            priors=priors if self.cfg.use_priors else None,
            assay_code=assay_feat
        )  # (B,)

        return pred


# ---------------------------------------------------------
# 7. minimal self-test (shape check)
# ---------------------------------------------------------
if __name__ == "__main__":
    # dummy batch to sanity check tensor shapes / forward
    cfg = ModelConfig()

    # Suppose batch size B=2
    B = 2
    # protein nodes total Np=5, ligand nodes total Nl=3
    Np = 5
    Nl = 3
    Ep = 8
    El = 6
    Ec = 4

    prot_pos = torch.randn(Np, 3)
    prot_x = torch.randn(Np, cfg.prot_in_dim)
    prot_edge_index = torch.randint(0, Np, (2, Ep))
    prot_edge_attr = torch.randn(Ep, cfg.edge_dim)
    prot_batch = torch.tensor([0,0,0,1,1], dtype=torch.long)

    lig_pos = torch.randn(Nl, 3)
    lig_x = torch.randn(Nl, cfg.lig_in_dim)
    lig_edge_index = torch.randint(0, Nl, (2, El))
    lig_edge_attr = torch.randn(El, cfg.edge_dim)
    lig_batch = torch.tensor([0,1,1], dtype=torch.long)

    # cross edges connect prot<->lig
    cross_index = torch.stack([
        torch.randint(0, Np, (Ec,)),
        torch.randint(0, Nl, (Ec,))
    ], dim=0)
    cross_attr = torch.randn(Ec, cfg.edge_dim)

    priors = torch.randn(B, cfg.priors_dim)

    model = BindingAffinityModel(cfg)
    out = model(
        prot_pos, prot_x, prot_edge_index, prot_edge_attr, prot_batch,
        lig_pos, lig_x, lig_edge_index, lig_edge_attr, lig_batch,
        cross_index, cross_attr,
        priors=priors,
        assay=None
    )
    print("pred shape:", out.shape)  # expect (B,)
