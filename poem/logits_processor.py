"""The PoEM logits processor for vLLM's V1 engine (Eq. 6).

A decode group is the set of requests that decode one prompt under one PoEM policy: one request per member
(adapter m or π_ref, weight w_m), with weights that sum to 1. Each request carries its group's name, its weight and
the group size in `SamplingParams.extra_args`. At every step the processor replaces each row of a complete group by
the fused logits Σ_m w_m ℓ_m, computed in fp32 and written back in the logits dtype, so all rows of the group pick the
same token. With the weights γα_k on the experts and 1 − γΣ_k α_k on π_ref this is Eq. 6 at composition
strength γ. Weights may be negative or above 1.

Groups differ in size, so each is padded to the largest with its own first row at weight 0. A group whose rows are
not all in the batch is left as it is; `sampling.decode_groups` adds whole groups in waves that are scheduled at once,
and counts the groups whose rows lost lockstep. This module subclasses a vLLM class, so importing it imports vLLM.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch
from vllm.v1.sample.logits_processor.builtin import process_dict_updates
from vllm.v1.sample.logits_processor.interface import BatchUpdate, LogitsProcessor

GROUP_KEY, WEIGHT_KEY, SIZE_KEY = "poem_group", "poem_weight", "poem_group_size"
SEEDED_KEY = "poem_seeded"      # a sampled group: its rows share one seed, so they draw the same tokens


@dataclass(frozen=True)
class _Member:
    group: str
    weight: float
    size: int


class PoEMLogitsProcessor(LogitsProcessor):
    """Replaces every row of each complete decode group by Σ_m w_m ℓ_m (fp32)."""

    def __init__(self, vllm_config, device, is_pin_memory):
        self._device = device
        self._members: dict[int, _Member] = {}
        self._groups: list[list[tuple[int, float]]] = []
        self._rows = torch.empty((0, 0), dtype=torch.long, device=device)
        self._w = torch.empty((0, 0, 1), dtype=torch.float32, device=device)
        self._max_row, self._cached_rows = -1, -1

    @classmethod
    def validate_params(cls, sp):
        """A group decodes greedily, or samples with one seed shared by its rows."""
        extra = sp.extra_args or {}
        if GROUP_KEY in extra and sp.temperature != 0.0 and not (extra.get(SEEDED_KEY) and sp.seed is not None):
            raise ValueError("PoEM groups require greedy decoding or a shared seed")

    @staticmethod
    def _new_member(params, _prompt_ids, _output_ids):
        extra = params.extra_args or {}
        if GROUP_KEY not in extra:
            return None
        return _Member(str(extra[GROUP_KEY]), float(extra[WEIGHT_KEY]), int(extra[SIZE_KEY]))

    def is_argmax_invariant(self):
        return False

    def update_state(self, batch_update: BatchUpdate | None):
        """Track the batch row of each member and keep the complete groups, rows sorted."""
        if not process_dict_updates(self._members, batch_update, self._new_member):
            return
        groups: dict[str, list] = defaultdict(list)
        sizes = {}
        for row, m in self._members.items():
            groups[m.group].append((row, m.weight))
            sizes[m.group] = m.size
        complete = []
        for g, members in groups.items():
            if len(members) != sizes[g]:
                continue                # a partly scheduled group stays uncomposed
            members.sort()
            s = sum(w for _, w in members)
            if abs(s - 1.0) > 1e-4:
                raise ValueError(f"PoEM group {g!r} weights sum to {s}, expected 1")
            complete.append(members)
        self._groups = complete
        self._max_row = max((max(r for r, _ in grp) for grp in complete), default=-1)
        self._cached_rows = -1
        self._build(complete)

    def _build(self, groups):
        """The (G, width) row indices and (G, width, 1) fp32 weights of the groups, padded to the largest group."""
        if not groups:
            self._rows = torch.empty((0, 0), dtype=torch.long, device=self._device)
            self._w = torch.empty((0, 0, 1), dtype=torch.float32, device=self._device)
            self._dst = self._gid = torch.empty((0,), dtype=torch.long, device=self._device)
            return
        width = max(len(g) for g in groups)
        rows = [[r for r, _ in g] + [g[0][0]] * (width - len(g)) for g in groups]
        ws = [[w for _, w in g] + [0.0] * (width - len(g)) for g in groups]
        self._rows = torch.tensor(rows, dtype=torch.long, device=self._device)
        self._w = torch.tensor(ws, dtype=torch.float32, device=self._device).unsqueeze(2)
        # the real rows and the group of each, for one write-back
        self._dst = torch.tensor([r for g in groups for r, _ in g], dtype=torch.long, device=self._device)
        self._gid = torch.tensor([i for i, g in enumerate(groups) for _ in g], dtype=torch.long, device=self._device)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._members or self._rows.numel() == 0:
            return logits
        n = logits.shape[0]
        if self._max_row >= n:          # fewer rows than the batch holds: fuse only the groups inside them
            if self._cached_rows != n:
                self._build([g for g in self._groups if all(r < n for r, _ in g)])
                self._cached_rows = n
            if self._rows.numel() == 0:
                return logits
        elif self._cached_rows != -1:
            self._build(self._groups)
            self._cached_rows = -1
        sel = logits.index_select(0, self._rows.flatten()).view(*self._rows.shape, logits.shape[1]).float()
        mixed = (sel * self._w).sum(dim=1)                              # (G, V) fp32
        logits.index_copy_(0, self._dst, mixed.index_select(0, self._gid).to(logits.dtype))
        return logits
