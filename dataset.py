import os
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _to_float32(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32)


def _to_int64(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.int64)


class PoseDataset(Dataset):
    """
    Load per-pose .npz graphs and expose tensors for training.

    Index JSONL format (flexible keys supported):
      - npz_path or path: string path to .npz
      - entry_id: string
      - pose_idx: int
      - all_passed: optional {0,1}

    filter_all_passed:
      - None  -> use all
      - True  -> keep only all_passed==1
      - False -> keep only all_passed==0
    """

    def __init__(self, index_file: str, filter_all_passed: Optional[bool] = None) -> None:
        assert os.path.exists(index_file), f"Index not found: {index_file}"
        self.records: List[Dict[str, Any]] = []
        with open(index_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # normalize keys
                npz_path = obj.get("npz_path") or obj.get("path") or obj.get("npz")
                if npz_path is None:
                    # Skip malformed entries silently
                    continue
                rec = {
                    "npz_path": npz_path,
                    "entry_id": obj.get("entry_id", ""),
                    "pose_idx": obj.get("pose_idx", 0),
                    "all_passed": obj.get("all_passed", 1),
                }
                self.records.append(rec)

        if filter_all_passed is not None:
            want = 1 if filter_all_passed else 0
            self.records = [r for r in self.records if r.get("all_passed", 1) == want]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        path = rec["npz_path"]
        # Note: In DDP, non-rank0 workers may hit this; consider try/except or rank-aware skipping
        if not os.path.exists(path):
            raise FileNotFoundError(f"NPZ not found: {path}")
        # structured dtype일 경우 pickle=True로 바꿀 수 있음
        data = np.load(path, allow_pickle=False)

        # Map file keys -> standardized output keys per spec
        # Protein
        prot_pos = _to_float32(data["prot_block_pos"])  # [N_prot,3]
        prot_x = _to_float32(data["prot_block_x"])      # [N_prot,D]
        prot_res_index = _to_int64(data["prot_block_res_index"])  # [N_prot]
        prot_edge_index = _to_int64(data["prot_block_edge_index"])  # [2,E_pp]
        prot_edge_attr = _to_float32(data["prot_block_edge_attr"])  # [E_pp,D]

        # Ligand
        lig_pos = _to_float32(data["lig_block_pos"])  # [N_lig,3]
        lig_x = _to_float32(data["lig_block_x"])      # [N_lig,D]

        # lig_block_members stored as flattened indices + lengths
        lig_members = None
        lig_members_lengths = None
        if "lig_block_members_lengths" in data:
            lig_members = _to_int64(data["lig_block_members"])  # [sum(len_i)]
            lig_members_lengths = _to_int64(data["lig_block_members_lengths"])  # [N_lig]
        else:
            # Backward compatibility: not present
            lig_members = torch.zeros((0,), dtype=torch.int64)
            lig_members_lengths = torch.zeros((0,), dtype=torch.int64)

        lig_edge_index = _to_int64(data["lig_block_edge_index"])  # [2,E_ff]
        lig_edge_attr = _to_float32(data["lig_block_edge_attr"])  # [E_ff,D]

        # Cross
        cross_index = _to_int64(data["cross_block_index"])  # [2,E_pl] (prot_idx, lig_idx)
        cross_dist = _to_float32(data["cross_block_dist"])  # [E_pl,D]

        # Targets & meta
        y = _to_float32(data["y"]).view(-1)  # [1] - collate_fn에서 [B,1]로 cat
        assay = _to_int64(data["assay"]).reshape(1, -1)  # [1,1]
        priors = _to_float32(data.get("priors", np.zeros((1,), dtype=np.float32))).reshape(1, -1)
        all_passed = torch.tensor([[rec.get("all_passed", 1)]], dtype=torch.int64)  # [1,1]

        entry_id = str(rec.get("entry_id", ""))
        pose_idx = torch.tensor([[int(rec.get("pose_idx", 0))]], dtype=torch.int64)  # [1,1]

        return {
            "prot_pos": prot_pos,
            "prot_x": prot_x,
            "prot_res_index": prot_res_index,
            "prot_edge_index": prot_edge_index,
            "prot_edge_attr": prot_edge_attr,

            "lig_pos": lig_pos,
            "lig_x": lig_x,
            "lig_members": lig_members,
            "lig_members_lengths": lig_members_lengths,
            "lig_edge_index": lig_edge_index,
            "lig_edge_attr": lig_edge_attr,

            "cross_index": cross_index,
            "cross_dist": cross_dist,

            "y": y,
            "assay": assay,
            "priors": priors,
            "all_passed": all_passed,

            "entry_id": entry_id,
            "pose_idx": pose_idx,
        }


def pose_collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Batch a list of pose dicts into a single graph-batch dict.

    - Concatenate node-wise tensors
    - Offset edge indices
    - Build prot_batch / lig_batch
    - Keep entry_id as list[str]
    - y, priors, assay, all_passed, pose_idx -> [B,1]
    """
    B = len(samples)

    # Accumulators
    prot_pos_list: List[torch.Tensor] = []
    prot_x_list: List[torch.Tensor] = []
    prot_res_index_list: List[torch.Tensor] = []
    prot_edge_index_list: List[torch.Tensor] = []
    prot_edge_attr_list: List[torch.Tensor] = []
    prot_batch_list: List[torch.Tensor] = []

    lig_pos_list: List[torch.Tensor] = []
    lig_x_list: List[torch.Tensor] = []
    lig_members_list: List[torch.Tensor] = []
    lig_members_lengths_list: List[torch.Tensor] = []
    lig_edge_index_list: List[torch.Tensor] = []
    lig_edge_attr_list: List[torch.Tensor] = []
    lig_batch_list: List[torch.Tensor] = []

    cross_index_list: List[torch.Tensor] = []
    cross_dist_list: List[torch.Tensor] = []

    y_list: List[torch.Tensor] = []
    assay_list: List[torch.Tensor] = []
    priors_list: List[torch.Tensor] = []
    all_passed_list: List[torch.Tensor] = []
    pose_idx_list: List[torch.Tensor] = []
    entry_ids: List[str] = []

    prot_offset = 0
    lig_offset = 0

    for b, s in enumerate(samples):
        Np = s["prot_pos"].shape[0]
        Nl = s["lig_pos"].shape[0]

        # Nodes
        prot_pos_list.append(s["prot_pos"])  # [Np,3]
        prot_x_list.append(s["prot_x"])      # [Np,D]
        prot_res_index_list.append(s["prot_res_index"])  # [Np]
        prot_batch_list.append(torch.full((Np,), b, dtype=torch.int64))

        lig_pos_list.append(s["lig_pos"])    # [Nl,3]
        lig_x_list.append(s["lig_x"])        # [Nl,D]
        lig_batch_list.append(torch.full((Nl,), b, dtype=torch.int64))

        # Ligand members (flattened + lengths)
        if s["lig_members_lengths"].numel() > 0:
            # Adjust atom indices inside members? They index original ligand atoms; no offset needed here
            lig_members_list.append(s["lig_members"])  # [sum(len_i)]
            lig_members_lengths_list.append(s["lig_members_lengths"])  # [Nl]
        else:
            lig_members_list.append(torch.zeros((0,), dtype=torch.int64))
            lig_members_lengths_list.append(torch.zeros((0,), dtype=torch.int64))

        # Edges (offset indices)
        if s["prot_edge_index"].numel() > 0:
            ei = s["prot_edge_index"] + prot_offset
            prot_edge_index_list.append(ei)
            prot_edge_attr_list.append(s["prot_edge_attr"])  # [Epp,D]
        if s["lig_edge_index"].numel() > 0:
            ei = s["lig_edge_index"] + lig_offset
            lig_edge_index_list.append(ei)
            lig_edge_attr_list.append(s["lig_edge_attr"])    # [Eff,D]

        if s["cross_index"].numel() > 0:
            ei = s["cross_index"].clone()
            ei = torch.vstack((ei[0] + prot_offset, ei[1] + lig_offset))
            cross_index_list.append(ei)
            cross_dist_list.append(s["cross_dist"])          # [Epl,D]

        # Targets & meta
        y_list.append(s["y"])                # [1] - will be unsqueezed to [B,1]
        assay_list.append(s["assay"])        # [1,1]
        priors_list.append(s["priors"])      # [1,P]
        all_passed_list.append(s["all_passed"])  # [1,1]
        pose_idx_list.append(s["pose_idx"])      # [1,1]
        entry_ids.append(s["entry_id"])      # str

        prot_offset += Np
        lig_offset += Nl

    # Concatenate
    out: Dict[str, Any] = {
        "prot_pos": torch.cat(prot_pos_list, dim=0) if prot_pos_list else torch.zeros((0, 3), dtype=torch.float32),
        "prot_x": torch.cat(prot_x_list, dim=0) if prot_x_list else torch.zeros((0, 23), dtype=torch.float32),
        "prot_res_index": torch.cat(prot_res_index_list, dim=0) if prot_res_index_list else torch.zeros((0,), dtype=torch.int64),
        "prot_edge_index": torch.cat(prot_edge_index_list, dim=1) if prot_edge_index_list else torch.zeros((2, 0), dtype=torch.int64),
        "prot_edge_attr": torch.cat(prot_edge_attr_list, dim=0) if prot_edge_attr_list else torch.zeros((0, 16), dtype=torch.float32),
        "prot_batch": torch.cat(prot_batch_list, dim=0) if prot_batch_list else torch.zeros((0,), dtype=torch.int64),

        "lig_pos": torch.cat(lig_pos_list, dim=0) if lig_pos_list else torch.zeros((0, 3), dtype=torch.float32),
        "lig_x": torch.cat(lig_x_list, dim=0) if lig_x_list else torch.zeros((0, 771), dtype=torch.float32),
        "lig_members": torch.cat(lig_members_list, dim=0) if lig_members_list else torch.zeros((0,), dtype=torch.int64),
        "lig_members_lengths": torch.cat(lig_members_lengths_list, dim=0) if lig_members_lengths_list else torch.zeros((0,), dtype=torch.int64),
        "lig_edge_index": torch.cat(lig_edge_index_list, dim=1) if lig_edge_index_list else torch.zeros((2, 0), dtype=torch.int64),
        "lig_edge_attr": torch.cat(lig_edge_attr_list, dim=0) if lig_edge_attr_list else torch.zeros((0, 16), dtype=torch.float32),
        "lig_batch": torch.cat(lig_batch_list, dim=0) if lig_batch_list else torch.zeros((0,), dtype=torch.int64),

        "cross_index": torch.cat(cross_index_list, dim=1) if cross_index_list else torch.zeros((2, 0), dtype=torch.int64),
        "cross_dist": torch.cat(cross_dist_list, dim=0) if cross_dist_list else torch.zeros((0, 16), dtype=torch.float32),

        "y": torch.cat(y_list, dim=0).unsqueeze(-1) if y_list else torch.zeros((0, 1), dtype=torch.float32),
        "assay": torch.cat(assay_list, dim=0) if assay_list else torch.zeros((0, 1), dtype=torch.int64),
        "priors": torch.cat(priors_list, dim=0) if priors_list else torch.zeros((0, 1), dtype=torch.float32),
        "all_passed": torch.cat(all_passed_list, dim=0) if all_passed_list else torch.zeros((0, 1), dtype=torch.int64),
        "pose_idx": torch.cat(pose_idx_list, dim=0) if pose_idx_list else torch.zeros((0, 1), dtype=torch.int64),

        "entry_id": entry_ids,
    }

    return out


