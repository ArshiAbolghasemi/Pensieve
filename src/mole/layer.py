import logging
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from config.mole import MoLELoRAConfig
from service.adapter import DecompositionMethod, SVDAdapterInitializer
from service.router import svd_router_initialization

logger = logging.getLogger(__name__)


class MoELoRALayer(nn.Module):
    """Mixture of Experts LoRA Layer compatible with PEFT."""

    def __init__(
        self,
        base_layer: nn.Linear,
        config: MoLELoRAConfig,
        layer_name: str,
        adapter_name: str = "default",
    ) -> None:
        super().__init__()
        logger.info(f"Initializing MoELoRALayer: {layer_name}")

        self.base_layer = base_layer
        self.config = config
        self.layer_name = layer_name
        self.adapter_name = adapter_name

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = config.r
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.lora_alpha = config.lora_alpha
        self.lora_dropout = config.lora_dropout
        self.scaling = self.lora_alpha / self.r

        self.residual_weight = None
        self.last_topk_indices = None
        self.last_expert_outputs = None

        self.router = self._initialize_router()
        self.lora_A_experts, self.lora_B_experts = self._initialize_experts()

        self.dropout = (
            nn.Dropout(p=self.lora_dropout) if self.lora_dropout > 0 else nn.Identity()
        )

        self.merged = False
        self.disable_adapters = False

    def _initialize_router(self) -> nn.Linear:
        logger.info(f"Initializing router for layer {self.layer_name}")
        router = nn.Linear(self.in_features, self.num_experts, bias=False)

        try:
            if self.config.router_init == "random":
                nn.init.normal_(router.weight, mean=0.0, std=0.01)
                logger.info("Router initialized with random normal weights")

            elif self.config.router_init == "orthogonal":
                with torch.no_grad():
                    nn.init.orthogonal_(router.weight)
                logger.info("Router initialized with orthogonal weights")

            elif self.config.router_init == "svd":
                R = svd_router_initialization(
                    weights=self.base_layer,
                    num_experts=self.num_experts,
                    in_dim=self.in_features,
                )
                with torch.no_grad():
                    router.weight.copy_(R.to(router.weight.dtype))
                logger.info("Router initialized using SVD-based initialization")

            else:
                raise ValueError(f"Unknown router_init: {self.config.router_init}")

        except Exception as e:
            logger.error(f"Router initialization failed: {e}")
            raise

        return router.to(self.base_layer.weight.device)

    def _initialize_experts(self) -> tuple[nn.ParameterList, nn.ParameterList]:
        logger.info(f"Initializing LoRA experts for layer {self.layer_name}")
        if self.config.adapter_init == "random":
            logger.info("Using random adapter initialization")
            return self._initialize_experts_random()

        logger.info(f"Using SVD-based adapter initialization: {self.config.adapter_init}")
        return self._initialize_experts_svd()

    def _initialize_experts_random(self) -> tuple[nn.ParameterList, nn.ParameterList]:
        lora_A_list = nn.ParameterList()
        lora_B_list = nn.ParameterList()

        device = self.base_layer.weight.device
        dtype = self.base_layer.weight.dtype

        for _ in range(self.num_experts):
            lora_A = nn.Parameter(
                torch.empty(self.r, self.in_features, device=device, dtype=dtype)
            )
            nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
            lora_B = nn.Parameter(
                torch.zeros(self.out_features, self.r, device=device, dtype=dtype)
            )
            lora_A_list.append(lora_A)
            lora_B_list.append(lora_B)
        return lora_A_list, lora_B_list

    def _initialize_experts_svd(self) -> tuple[nn.ParameterList, nn.ParameterList]:
        method_map = {
            "pissa": DecompositionMethod.PISSA,
            "milora": DecompositionMethod.MILORA,
            "goat": DecompositionMethod.GOAT,
            "goat_mini": DecompositionMethod.GOAT_MINI,
            "pissa_milora": DecompositionMethod.PISSA_MILORA,
        }
        method = method_map[self.config.adapter_init]

        device = self.base_layer.weight.device
        dtype = self.base_layer.weight.dtype

        if not torch.is_floating_point(self.base_layer.weight):
            dtype = torch.float32

        with torch.no_grad():
            initializer = SVDAdapterInitializer(
                layer=self.base_layer,
                rank=self.r,
                method=method,
                num_experts=self.num_experts,
                scaling_factor=1.0,
                init_coefficient=1.0,
                rho=10.0,
            )
            lora_A_list, lora_B_list, residual_weight = initializer.execute()

            if residual_weight is not None:
                self.residual_weight = residual_weight.to(device=device, dtype=dtype)

        lora_A_params = nn.ParameterList(
            [nn.Parameter(a.to(device=device, dtype=dtype)) for a in lora_A_list]
        )
        lora_B_params = nn.ParameterList(
            [nn.Parameter(b.to(device=device, dtype=dtype)) for b in lora_B_list]
        )

        return lora_A_params, lora_B_params

    def compute_diversity_loss(self, topk_indices: Tensor) -> Tensor:
        """Compute diversity loss for top-k selected experts with memory-efficient implementation."""
        device = topk_indices.device

        if self.top_k <= 1:
            return torch.zeros((), device=device)

        batch_size = topk_indices.shape[0]

        # Limit samples to prevent OOM
        max_samples = 256
        if batch_size > max_samples:
            sample_indices = torch.randperm(batch_size, device=device)[:max_samples]
            topk_indices = topk_indices[sample_indices]
            batch_size = max_samples

        # Compute expert norms
        norms_squared = torch.zeros(self.num_experts, device=device)

        for expert_idx in range(self.num_experts):
            A_expert = self.lora_A_experts[expert_idx]
            B_expert = self.lora_B_experts[expert_idx]
            BtB = torch.matmul(B_expert.transpose(0, 1), B_expert)
            temp = torch.matmul(BtB, A_expert)
            norms_squared[expert_idx] = torch.sum(A_expert * temp)

        norms = torch.sqrt(norms_squared + 1e-8)

        # Compute similarities
        total_similarity = torch.tensor(0.0, device=device, requires_grad=True)
        num_comparisons = 0

        for i in range(self.top_k):
            for j in range(i + 1, self.top_k):
                idx_i = topk_indices[:, i]
                idx_j = topk_indices[:, j]

                unique_pairs = torch.unique(torch.stack([idx_i, idx_j], dim=1), dim=0)
                pair_similarities = []

                for pair_idx in range(unique_pairs.shape[0]):
                    expert_i = unique_pairs[pair_idx, 0].item()
                    expert_j = unique_pairs[pair_idx, 1].item()

                    A_i = self.lora_A_experts[expert_i]
                    B_i = self.lora_B_experts[expert_i]
                    A_j = self.lora_A_experts[expert_j]
                    B_j = self.lora_B_experts[expert_j]

                    BiBj = torch.matmul(B_i.transpose(0, 1), B_j)
                    temp = torch.matmul(BiBj, A_j)
                    inner_product = torch.sum(A_i * temp)

                    norm_i = norms[expert_i]
                    norm_j = norms[expert_j]
                    cosine_sim = inner_product / (norm_i * norm_j + 1e-8)

                    pair_count = ((idx_i == expert_i) & (idx_j == expert_j)).sum()
                    pair_similarities.append(cosine_sim**2 * pair_count)
                    num_comparisons += pair_count.item()

                if pair_similarities:
                    total_similarity = (
                        total_similarity + torch.stack(pair_similarities).sum()
                    )

        if num_comparisons > 0:
            return total_similarity / num_comparisons
        return total_similarity

    def compute_expert_outputs(self, x: Tensor) -> Tensor:
        """Compute outputs for all experts without routing."""
        batch_size = x.shape[0]
        expert_outputs = torch.zeros(
            batch_size, self.num_experts, self.out_features, device=x.device, dtype=x.dtype
        )

        for expert_id in range(self.num_experts):
            lora_A = self.lora_A_experts[expert_id]
            lora_B = self.lora_B_experts[expert_id]
            lora_out = (x @ lora_A.T) @ lora_B.T
            lora_out = lora_out * self.scaling
            expert_outputs[:, expert_id, :] = lora_out

        return expert_outputs

    def forward(self, x: Tensor) -> Tensor:
        # If adapters are disabled, just use base layer
        if self.disable_adapters or self.merged:
            return self.base_layer(x)

        original_shape = x.shape
        x_flat = x.view(-1, self.in_features)

        # Router forward pass
        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)

        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        self.last_topk_indices = topk_indices.detach()

        # Compute expert outputs for metrics (without gradients)
        with torch.no_grad():
            self.last_expert_outputs = self.compute_expert_outputs(x_flat)

        # Compute weighted expert outputs
        expert_output = torch.zeros(
            x_flat.shape[0], self.out_features, device=x.device, dtype=x.dtype
        )

        for k_idx in range(self.top_k):
            expert_idx = topk_indices[:, k_idx]
            expert_weight = topk_probs[:, k_idx : k_idx + 1]

            for expert_id in range(self.num_experts):
                mask = expert_idx == expert_id
                if not mask.any():
                    continue

                x_expert = x_flat[mask]
                weight_expert = expert_weight[mask]

                lora_A = self.lora_A_experts[expert_id]
                lora_B = self.lora_B_experts[expert_id]

                x_dropped = self.dropout(x_expert)
                lora_out = (x_dropped @ lora_A.T) @ lora_B.T
                lora_out = lora_out * self.scaling
                expert_output[mask] += weight_expert * lora_out

        expert_output = expert_output.view(*original_shape[:-1], self.out_features)

        base_output = self.base_layer(x)

        if self.residual_weight is not None:
            residual_output = F.linear(x, self.residual_weight, None)
            final_output = base_output - residual_output + expert_output
        else:
            final_output = base_output + expert_output

        return final_output
