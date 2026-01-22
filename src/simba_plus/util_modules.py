from typing import Dict, Tuple, List
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData


class BufferDict(nn.Module):
    def __init__(self, d: Dict[str, Tensor]):
        super().__init__()
        for k, v in d.items():
            name = f"_{k}"
            self.register_buffer(name, v)
    
    def __getitem__(self, key: str) -> Tensor:
        return getattr(self, f"_{key}")


class _CheckHeteroDataCodes:
    """
    Check if the codes are consecutive and consistent (sample/batch/batch_local).
    Optionally generate missing `cell.batch` or `cell.batch_local` in-place.
    """

    @classmethod
    def _as_sorted_unique_list(cls, x: torch.Tensor) -> List[int]:
        if x is None or x.numel() == 0:
            return []
        return sorted(torch.unique(x).detach().cpu().tolist())

    @classmethod
    def _is_subset(cls, ref: torch.Tensor, sub: torch.Tensor) -> bool:
        ref_u, sub_u = cls._as_sorted_unique_list(ref), cls._as_sorted_unique_list(sub)
        return set(sub_u) <= set(ref_u)

    @classmethod
    def check_consecutive(cls, codes: torch.Tensor, start_with: int = 0, allow_empty: bool = False) -> bool:
        uniq = cls._as_sorted_unique_list(codes)
        if len(uniq) == 0:
            return allow_empty
        expected = list(range(start_with, start_with + len(uniq)))
        return uniq == expected

    @classmethod
    def _remap_to_consecutive(cls, codes: torch.Tensor, start_with: int = 0) -> torch.Tensor:
        """
        Remap arbitrary integer codes to consecutive codes starting at start_with.
        Returned tensor is same shape as codes.
        """
        if codes.numel() == 0:
            return codes.clone()
        uniq = torch.unique(codes)
        uniq_sorted = torch.sort(uniq).values
        # map each code -> its index in uniq_sorted
        # safe because uniq_sorted contains all codes
        idx = torch.searchsorted(uniq_sorted, codes)
        return idx.to(torch.long) + int(start_with)
    
    @classmethod
    def _make_batch_local_from_global(
        cls,
        sample_codes: torch.Tensor,   # (N,)
        batch_codes: torch.Tensor,    # (N,) global batch codes (may be non-consecutive)
    ) -> torch.Tensor:
        """
        For each sample, remap its global batch codes to local consecutive codes 0..B_s-1.

        Returns:
            batch_local: (N,) long tensor
        """
        if sample_codes.ndim != 1 or batch_codes.ndim != 1:
            raise ValueError("sample_codes and batch_codes must be 1D")
        if sample_codes.numel() != batch_codes.numel():
            raise ValueError("sample_codes and batch_codes must have the same length")
        if batch_codes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=batch_codes.device)

        device = batch_codes.device
        out = torch.empty_like(batch_codes, dtype=torch.long, device=device)

        # iterate samples in sorted unique order (stable)
        for sid in cls._as_sorted_unique_list(sample_codes):
            sid = int(sid)
            mask = (sample_codes == sid)
            b = batch_codes[mask]
            # remap b to 0..K-1 (consecutive)
            local = cls._remap_to_consecutive(b)
            out[mask] = local

        return out

    @classmethod
    def _make_batch_global_from_local(
        cls,
        sample_codes: torch.Tensor,     # (N,)
        batch_local: torch.Tensor,      # (N,) local batch codes per sample (may be non-consecutive)
    ) -> torch.Tensor:
        """
        Build a global consecutive batch code by offsetting each sample's local batches.

        For each sample s:
        - remap its batch_local values to consecutive 0..B_s-1 (even if input isn't)
        - then add an offset that accumulates previous samples' B

        Returns:
            batch_global: (N,) long tensor, consecutive 0..(sum B_s - 1)
        """
        if sample_codes.ndim != 1 or batch_local.ndim != 1:
            raise ValueError("sample_codes and batch_local must be 1D")
        if sample_codes.numel() != batch_local.numel():
            raise ValueError("sample_codes and batch_local must have the same length")
        if batch_local.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=sample_codes.device)

        device = batch_local.device
        out = torch.empty_like(batch_local, dtype=torch.long, device=device)

        offset = 0
        for sid in cls._as_sorted_unique_list(sample_codes):
            sid = int(sid)
            mask = (sample_codes == sid)
            bl = batch_local[mask]
            blc = cls._remap_to_consecutive(bl)
            out[mask] = blc + offset
            offset += int(blc.max()) + 1 if blc.numel() > 0 else 0

        return out

    @classmethod
    def check_heterodata(
        cls,
        data: HeteroData,
        start_with: int = 0,
        allow_empty: bool = False,
    ) -> Tuple[bool, List[str]]:
        ok, msgs = True, []

        # ---- basic structure checks ----
        if "cell" not in data.node_types:
            ok = False
            msgs.append("Cell node type is required")

        feature_types = list(set(data.node_types) - {"cell"})
        if len(feature_types) == 0:
            ok = False
            msgs.append("At least one feature node type is required")

        if not ok:
            return False, msgs

        # ---- sample checks ----
        if hasattr(data["cell"], "sample"):
            sample_codes = data["cell"].sample
            if not cls.check_consecutive(sample_codes, start_with, allow_empty):
                ok = False
                msgs.append(
                    f"Sample codes for cell node type are not consecutive (expected {start_with}..K{start_with-1 if start_with <= 0 else ('' if start_with == 1 else '+'+str(start_with-1))}). "
                    f"Got unique={cls._as_sorted_unique_list(sample_codes)[:20]}"
                )

            # feature sample subset check
            for node_type in feature_types:
                if not hasattr(data[node_type], "sample"):
                    ok = False
                    msgs.append(f"Sample attribute is required for node type {node_type}")
                    continue
                if not cls._is_subset(sample_codes, data[node_type].sample):
                    ok = False
                    msgs.append(
                        f"Sample codes for node type {node_type} are not a subset of cell sample codes. "
                        f"{node_type} unique (first 20) = {cls._as_sorted_unique_list(data[node_type].sample)[:20]}, "
                        f"cell unique (first 20) = {cls._as_sorted_unique_list(sample_codes)[:20]}"
                    )
        else:
            # If no sample codes, we treat as single-sample data.
            sample_codes = torch.zeros(data["cell"].num_nodes, dtype=torch.long)
        
        if not ok:
            return False, msgs

        # ---- batch / batch_local checks or generation ----
        has_batch = hasattr(data["cell"], "batch")
        has_batch_local = hasattr(data["cell"], "batch_local")

        if has_batch:
            batch = data["cell"].batch
            if not cls.check_consecutive(batch, start_with, allow_empty):
                ok = False
                msgs.append(
                    f"Batch codes for cell node type are not consecutive (global). First 20 unique = {cls._as_sorted_unique_list(batch)[:20]}"
                )
            
            if not ok:
                return False, msgs

            if not has_batch_local:
                # generate batch_local based on batch and sample codes
                data["cell"].batch_local = cls._make_batch_local_from_global(sample_codes, batch)
                msgs.append("Generated cell.batch_local from cell.batch since cell.batch_local is missing")

            else:
                # check if batch_local is correct based on batch and sample codes
                batch_local = cls._make_batch_local_from_global(sample_codes, batch)
                if not torch.equal(data["cell"].batch_local, batch_local):
                    ok = False
                    msgs.append(
                        f"Inconsistent cell.batch_local vs cell.batch")
        else:
            if has_batch_local:
                # check if batch_local is correct based on batch and sample codes
                batch_local = data["cell"].batch_local
                data["cell"].batch = cls._make_batch_global_from_local(sample_codes, batch_local)
                msgs.append("Generated cell.batch from cell.batch_local since cell.batch is missing")
        
        return ok, msgs

