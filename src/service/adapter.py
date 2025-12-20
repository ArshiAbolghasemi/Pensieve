import warnings
from enum import Enum

import bitsandbytes as bnb
import torch
from torch import nn


class DecompositionMethod(Enum):
    """Supported SVD decomposition methods"""

    PISSA = "pissa"
    MILORA = "milora"
    GOAT = "goat"
    GOAT_MINI = "goat_mini"
    PISSA_MILORA = "pissa_milora"


def get_svd_methods() -> list[str]:
    return [m.value for m in DecompositionMethod]


class SVDAdapterInitializer:
    """Initialize low-rank adapters (LoRA-like matrices) for Mixture-of-Experts (MoE)
    using SVD-based decomposition of a pretrained layer weight.

    This class extracts the weight matrix from a given `nn.Linear` layer,
    performs Singular Value Decomposition (SVD), and uses configurable
    strategies (PISSA, MILORA, GOAT, etc.) to select and combine components.

    The resulting low-rank factors `A` and `B` can be used to initialize
    expert-specific LoRA adapters or parameter-efficient fine-tuning modules.

    ---
    Workflow:
        1. Decompose W = U Σ Vᵀ via SVD.
        2. Select subset of (U, Σ, Vᵀ) according to the chosen strategy.
        3. Construct low-rank factors:
              LoRA_B = U √Σ
              LoRA_A = √Σ Vᵀ
        4. Partition (A, B) across experts if `num_experts > 1`.
        5. Return (A_list, B_list, residual_weight)

    ---

    Args:
        layer (nn.Module):
            A linear layer whose weight matrix will be decomposed.
        rank (int):
            Target rank for low-rank decomposition.
        method (DecompositionMethod, optional):
            Strategy for selecting singular components (default: GOAT).
        num_experts (int, optional):
            Number of experts for partitioning LoRA adapters.
        scaling_factor (float, optional):
            Multiplier applied to control LoRA scaling.
        init_coefficient (float, optional):
            Coefficient controlling residual reconstruction.
        rho (float, optional):
            Damping constant for scaling singular values.

    ---

    Returns:
        tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
            - lora_A_list: Low-rank right factors (per expert)
            - lora_B_list: Low-rank left factors (per expert)
            - residual_weight: Remaining weight after removing low-rank part

    """

    def __init__(
        self,
        *,
        layer: nn.Module,
        rank: int,
        method: DecompositionMethod = DecompositionMethod.GOAT,
        num_experts: int = 1,
        scaling_factor: float = 1.0,
        init_coefficient: float = 1.0,
        rho: float = 10.0,
    ) -> None:
        if not isinstance(layer, nn.Linear):
            raise ValueError(f"Only nn.Linear layers supported, got {type(layer)}")

        self.layer = layer
        self.rank = rank
        self.method = method
        self.num_experts = num_experts
        self.scaling_factor = scaling_factor
        self.init_coefficient = init_coefficient
        self.rho = rho

        if rank <= 0:
            raise ValueError(f"Rank must be positive, got {rank}")
        if num_experts > 1 and rank % num_experts != 0:
            raise ValueError(f"Rank {rank} must be divisible by num_experts {num_experts}")

    def _get_weight_data(self) -> tuple[torch.Tensor, torch.dtype, torch.device]:
        """Extract weight data from layer, handling quantized weights.

        Returns:
            - weight: Dequantized weight tensor in float32
            - original_dtype: Original dtype of the weight
            - original_device: Original device of the weight

        """
        layer = self.layer

        # Check if layer is quantized (bitsandbytes)
        if hasattr(layer, "weight") and hasattr(layer.weight, "quant_state"):
            # Dequantize the weight
            quant_state = layer.weight.quant_state
            if quant_state is not None:
                # Use bitsandbytes dequantization
                weight = bnb.functional.dequantize_4bit(
                    layer.weight.data,
                    quant_state=quant_state,
                )
                original_dtype = (
                    torch.float16
                )  # Quantized weights are typically stored as fp16
                original_device = weight.device
            else:
                # Fallback if quant_state is None
                weight = layer.weight.data
                original_dtype = weight.dtype
                original_device = weight.device

        # Check for other quantization formats (e.g., torch.qint8)
        elif hasattr(layer, "weight") and layer.weight.dtype in [
            torch.qint8,
            torch.quint8,
            torch.qint32,
        ]:
            # Dequantize torch quantized tensors
            weight = torch.dequantize(layer.weight)
            original_dtype = torch.float32
            original_device = weight.device

        # Standard non-quantized layer
        else:
            weight = layer.weight.data
            original_dtype = weight.dtype
            original_device = weight.device

        return weight, original_dtype, original_device

    def execute(self) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        """Perform SVD decomposition and return low-rank components.

        Returns:
            - list of lora_A matrices (one per expert)
            - list of lora_B matrices (one per expert)
            - Updated residual weight matrix

        """
        # Get weight data (dequantize if needed)
        weight, original_dtype, original_device = self._get_weight_data()

        # Check for NaN or Inf values
        if torch.isnan(weight).any():
            raise ValueError("Weight matrix contains NaN values. Cannot perform SVD.")
        if torch.isinf(weight).any():
            raise ValueError("Weight matrix contains Inf values. Cannot perform SVD.")

        # Convert to float32 for numerical stability during SVD
        weight = weight.to(torch.float32)

        # Perform SVD with fallback to CPU if CUDA fails
        try:
            U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
        except (torch._C._LinAlgError, RuntimeError) as e:
            warnings.warn(
                f"CUDA SVD failed with error: {e}\n"
                "Falling back to CPU computation. This may be slower but more stable.",
                UserWarning,
            )
            weight_cpu = weight.cpu()
            U, S, Vh = torch.linalg.svd(weight_cpu, full_matrices=False)
            U = U.to(original_device)
            S = S.to(original_device)
            Vh = Vh.to(original_device)

        # Verify SVD results
        if torch.isnan(S).any() or (S < 0).any():
            raise ValueError(
                "SVD produced invalid singular values (NaN or negative). "
                "This indicates numerical instability in the weight matrix."
            )

        Ur, Sr, Vhr = self._select_components(U, S, Vh)

        rho = self.rho
        Sr = Sr / (self.scaling_factor * rho)

        # Check for very small singular values
        min_sv = Sr.min()
        if min_sv < 1e-10:
            warnings.warn(
                f"Very small singular values detected (min: {min_sv:.2e}). "
                "This may cause numerical instability. Consider increasing rho parameter.",
                UserWarning,
            )

        lora_A_full = torch.diag(torch.sqrt(Sr)) @ Vhr
        lora_B_full = Ur @ torch.diag(torch.sqrt(Sr))

        lora_A_list, lora_B_list = self._partition_into_experts(lora_A_full, lora_B_full)

        residual_weight = weight - self.init_coefficient * self.scaling_factor * (
            lora_B_full @ lora_A_full
        )
        residual_weight = residual_weight.to(original_dtype)
        residual_weight = residual_weight.to(dtype=original_dtype, device=original_device)

        lora_A_list = [a.to(original_dtype) for a in lora_A_list]
        lora_B_list = [b.to(original_dtype) for b in lora_B_list]

        return lora_A_list, lora_B_list, residual_weight

    def _select_components(
        self, U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select SVD components based on decomposition method."""
        if self.method == DecompositionMethod.PISSA:
            Ur = U[:, : self.rank]
            Sr = S[: self.rank]
            Vhr = Vh[: self.rank, :]

        elif self.method == DecompositionMethod.MILORA:
            Ur = U[:, -self.rank :]
            Sr = S[-self.rank :]
            Vhr = Vh[-self.rank :, :]

        elif self.method == DecompositionMethod.PISSA_MILORA:
            half_rank = self.rank // 2
            Ur = torch.cat((U[:, :half_rank], U[:, -half_rank:]), dim=1)
            Sr = torch.cat((S[:half_rank], S[-half_rank:]))
            Vhr = torch.cat((Vh[:half_rank, :], Vh[-half_rank:, :]), dim=0)

        elif self.method == DecompositionMethod.GOAT:
            Vlen = U.shape[1] // self.num_experts
            Mlen = self.rank // self.num_experts
            U_pieces = [U[:, i * Vlen : i * Vlen + Mlen] for i in range(self.num_experts)]
            S_pieces = [S[i * Vlen : i * Vlen + Mlen] for i in range(self.num_experts)]
            Vh_pieces = [Vh[i * Vlen : i * Vlen + Mlen, :] for i in range(self.num_experts)]
            Ur = torch.cat(U_pieces, dim=1)
            Sr = torch.cat(S_pieces)
            Vhr = torch.cat(Vh_pieces, dim=0)

        elif self.method == DecompositionMethod.GOAT_MINI:
            Vlen = U.shape[1] // self.num_experts
            Mlen = self.rank // self.num_experts
            U_pieces = [
                U[:, (i + 1) * Vlen - Mlen : (i + 1) * Vlen]
                for i in range(self.num_experts)
            ]
            S_pieces = [
                S[(i + 1) * Vlen - Mlen : (i + 1) * Vlen] for i in range(self.num_experts)
            ]
            Vh_pieces = [
                Vh[(i + 1) * Vlen - Mlen : (i + 1) * Vlen, :]
                for i in range(self.num_experts)
            ]
            Ur = torch.cat(U_pieces, dim=1)
            Sr = torch.cat(S_pieces)
            Vhr = torch.cat(Vh_pieces, dim=0)

        else:
            raise NotImplementedError(f"Method {self.method} not implemented")

        return Ur, Sr, Vhr

    def _partition_into_experts(
        self, lora_A: torch.Tensor, lora_B: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Partition full low-rank matrices into per-expert components."""
        rank_per_expert = self.rank // self.num_experts
        lora_A_list, lora_B_list = [], []

        for i in range(self.num_experts):
            start_idx = i * rank_per_expert
            end_idx = start_idx + rank_per_expert
            lora_A_list.append(lora_A[start_idx:end_idx, :].contiguous())
            lora_B_list.append(lora_B[:, start_idx:end_idx].contiguous())

        return lora_A_list, lora_B_list
