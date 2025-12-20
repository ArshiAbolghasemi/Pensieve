import warnings

import bitsandbytes as bnb
import torch
from torch import nn


def svd_router_initialization(
    weights: torch.Tensor | nn.Linear, num_experts: int, in_dim: int
) -> torch.Tensor:
    """Initialize Mixture-of-Experts (MoE) router weights using SVD.

    R_i = sqrt(σ_i) * v_i^T (using right singular vectors)

    Args:
        weights (torch.Tensor or nn.Linear): Weight matrix or linear layer.
        num_experts (int): Number of experts (router output dimension).
        in_dim (int): Input dimension (router input dimension).

    Returns:
        torch.Tensor: Router matrix of shape [num_experts, in_dim] (matches nn.Linear weight format).

    """
    # Extract weight if nn.Linear
    if isinstance(weights, nn.Linear):
        weight = weights.weight.data

        # bitsandbytes 4-bit dequantization
        if (
            hasattr(weights.weight, "quant_state")
            and weights.weight.quant_state is not None
        ):
            weight = bnb.functional.dequantize_4bit(
                weight, quant_state=weights.weight.quant_state
            )

        # torch quantized tensor
        elif weight.dtype in [torch.qint8, torch.quint8, torch.qint32]:
            weight = torch.dequantize(weight)
    else:
        weight = weights

    # Ensure weight is a tensor
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor or nn.Linear, got {type(weight)}")

    # Convert to float32 for SVD stability
    weight = weight.to(torch.float32)

    # Perform SVD (CPU fallback if CUDA fails)
    try:
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    except RuntimeError as e:
        warnings.warn(f"CUDA SVD failed: {e}, falling back to CPU.")
        weight_cpu = weight.cpu()
        U, S, Vh = torch.linalg.svd(weight_cpu, full_matrices=False)
        U = U.to(weight.device)
        S = S.to(weight.device)
        Vh = Vh.to(weight.device)

    # Use right singular vectors (Vh rows correspond to input space directions)
    # Vh has shape [min(out_features, in_features), in_features]
    # We need [num_experts, in_dim]

    # Crop to match desired number of experts
    rows = min(num_experts, Vh.shape[0])
    R = Vh[:rows, :] * S[:rows].sqrt().unsqueeze(1)

    # If num_experts > available rows, pad with zeros
    if num_experts > R.shape[0]:
        pad = torch.zeros(
            num_experts - R.shape[0], R.shape[1], device=R.device, dtype=R.dtype
        )
        R = torch.cat([R, pad], dim=0)

    # Ensure correct input dimension
    if R.shape[1] != in_dim:
        if R.shape[1] > in_dim:
            R = R[:, :in_dim]
        else:
            pad = torch.zeros(
                R.shape[0], in_dim - R.shape[1], device=R.device, dtype=R.dtype
            )
            R = torch.cat([R, pad], dim=1)

    return R
