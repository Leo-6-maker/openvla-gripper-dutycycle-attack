from __future__ import annotations

import hashlib
import io
from typing import Iterable, Sequence

import torch


def tensor_sha256(tensor: torch.Tensor) -> str:
    buffer = io.BytesIO()
    torch.save(tensor.detach().cpu(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def rand_seed_schedule(base_seed: int, count: int = 20) -> list[int]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(base_seed))
    return [int(x) for x in torch.randint(0, 2**31 - 1, (int(count),), generator=gen).tolist()]


def sample_processor_delta(
    shape: Sequence[int],
    *,
    epsilon: float,
    seed: int,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    delta = torch.empty(tuple(int(x) for x in shape), dtype=torch.float32, device="cpu")
    delta.uniform_(-float(epsilon), float(epsilon), generator=gen)
    return delta.to(device=device, dtype=dtype)


def project_and_cast_processor_values(
    x_orig: torch.Tensor,
    candidate: torch.Tensor,
    *,
    epsilon: float,
    candidate_is_delta: bool = True,
    tolerance: float = 1e-7,
) -> tuple[torch.Tensor, int]:
    """Project in fp32 and cast to ``x_orig`` dtype without budget overflow.

    The processor-space budget is audited on the actual tensor consumed by the
    model.  bf16/fp16 casts can round a projected fp32 boundary value outside
    ``x_orig +/- epsilon``; those elements are reset to the original value.
    This helper is intentionally shared by PGD and random controls.
    """

    x_orig_float = x_orig.detach().float()
    candidate_float = candidate.to(device=x_orig.device, dtype=torch.float32)
    adv_float = x_orig_float + candidate_float if candidate_is_delta else candidate_float
    projected = torch.max(
        torch.min(adv_float, x_orig_float + float(epsilon)),
        x_orig_float - float(epsilon),
    )
    if x_orig.dtype == torch.float32:
        return projected.to(device=x_orig.device), 0
    casted = projected.to(dtype=x_orig.dtype)
    over_budget = (casted.float() - x_orig_float).abs() > (float(epsilon) + float(tolerance))
    correction_count = int(torch.sum(over_budget).detach().cpu())
    if correction_count:
        casted = torch.where(over_budget, x_orig.detach(), casted)
    return casted, correction_count


def project_processor_space(x_orig: torch.Tensor, delta: torch.Tensor, *, epsilon: float) -> torch.Tensor:
    projected, _ = project_and_cast_processor_values(
        x_orig,
        delta,
        epsilon=float(epsilon),
        candidate_is_delta=True,
    )
    return projected


def select_best_surrogate_only(candidate_ids: Iterable[int], surrogate_scores: Sequence[float]) -> int:
    ids = [int(x) for x in candidate_ids]
    scores = [float(x) for x in surrogate_scores]
    if len(ids) != len(scores) or not ids:
        raise ValueError("candidate_ids and surrogate_scores must have the same non-zero length")
    best = max(range(len(ids)), key=lambda i: scores[i])
    return ids[best]


def shuffled_grad_direction(grad: torch.Tensor, *, seed: int, mode: str = "permute") -> torch.Tensor:
    flat = grad.detach().flatten()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    if str(mode) == "rademacher":
        signs = torch.randint(0, 2, flat.shape, generator=gen, dtype=torch.int64).float() * 2.0 - 1.0
        return signs.to(device=grad.device, dtype=grad.dtype).view_as(grad)
    perm = torch.randperm(flat.numel(), generator=gen)
    return flat.cpu()[perm].to(device=grad.device, dtype=grad.dtype).view_as(grad)
