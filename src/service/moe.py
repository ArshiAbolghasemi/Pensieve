import logging
import math
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import PreTrainedModel

from config.moe import MoELoRAConfig
from service.adapter import DecompositionMethod, SVDAdapterInitializer
from service.router import svd_router_initialization

logger = logging.getLogger(__name__)


class MoELoRALayer(nn.Module):
    """Mixture of Experts LoRA Layer with configurable initialization."""

    def __init__(
        self,
        base_layer: nn.Linear,
        config: MoELoRAConfig,
        layer_name: str,
    ) -> None:
        super().__init__()
        logger.info(f"Initializing MoELoRALayer: {layer_name}")

        self.base_layer = base_layer
        self.config = config
        self.layer_name = layer_name

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = config.r
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.lora_alpha = config.lora_alpha
        self.lora_dropout = config.lora_dropout
        self.scaling = self.lora_alpha / self.r

        self.router = self._initialize_router()
        self.lora_A_experts, self.lora_B_experts = self._initialize_experts()

        self.dropout = (
            nn.Dropout(p=self.lora_dropout) if self.lora_dropout > 0 else nn.Identity()
        )

        self.residual_weight = None
        self.last_topk_indices = None

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
        """Efficiently compute diversity loss using LOW-RANK similarity.

        Key insight: For matrices W_i = B_i @ A_i and W_j = B_j @ A_j,
        the Frobenius inner product is:
            <W_i, W_j> = Tr(W_i^T @ W_j) = Tr(A_i^T @ B_i^T @ B_j @ A_j)

        This can be computed in O(r^2 * (m + n)) instead of O(m * n)!

        Memory: O(r^2) instead of O(m * n)
        Speed: ~100-1000x faster for typical LoRA ranks
        """
        device = topk_indices.device

        if self.top_k <= 1:
            return torch.zeros((), device=device)

        batch_size = topk_indices.shape[0]

        max_samples = 1000
        if batch_size > max_samples:
            sample_indices = torch.randperm(batch_size, device=device)[:max_samples]
            topk_indices = topk_indices[sample_indices]
            batch_size = max_samples

        A_stack = torch.stack(
            [self.lora_A_experts[i] for i in range(self.num_experts)], dim=0
        )
        B_stack = torch.stack(
            [self.lora_B_experts[i] for i in range(self.num_experts)], dim=0
        )

        BtB = torch.bmm(B_stack.transpose(1, 2), B_stack)
        AtBtB = torch.bmm(A_stack, BtB)
        norms_squared = torch.einsum("eij,eji->e", A_stack, AtBtB)
        norms = torch.sqrt(norms_squared + 1e-8)

        total_similarity = torch.tensor(0.0, device=device)
        num_pairs = 0

        for i in range(self.top_k):
            for j in range(i + 1, self.top_k):
                idx_i = topk_indices[:, i]
                idx_j = topk_indices[:, j]

                A_i = A_stack[idx_i]
                B_i = B_stack[idx_i]
                A_j = A_stack[idx_j]
                B_j = B_stack[idx_j]

                BiBj = torch.bmm(B_i.transpose(1, 2), B_j)

                temp = torch.bmm(BiBj, A_j)
                inner_product = torch.einsum("brn,brn->b", A_i, temp)

                norm_i = norms[idx_i]
                norm_j = norms[idx_j]

                cosine_sim = inner_product / (norm_i * norm_j + 1e-8)

                total_similarity = total_similarity + cosine_sim.mean()
                num_pairs += 1

        return total_similarity / num_pairs if num_pairs > 0 else total_similarity

    def forward(self, x: Tensor) -> Tensor:
        original_shape = x.shape
        x_flat = x.view(-1, self.in_features)

        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)

        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        self.last_topk_indices = topk_indices

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


class MoELoRAModel(nn.Module):
    def __init__(self, base_model: PreTrainedModel, config: MoELoRAConfig) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.moe_layers: nn.ModuleDict = nn.ModuleDict()

        logger.info("Starting MoE layer injection")
        self._inject_moe_layers()
        logger.info(f"MoE injection completed | layers={len(self.moe_layers)}")

    def _inject_moe_layers(self) -> None:
        for name, module in self.base_model.named_modules():
            if any(t in name for t in self.config.target_modules) and isinstance(
                module, nn.Linear
            ):
                parent_name = ".".join(name.split(".")[:-1])
                attr_name = name.split(".")[-1]
                parent = (
                    self.base_model.get_submodule(parent_name)
                    if parent_name
                    else self.base_model
                )

                moe_layer = MoELoRALayer(module, self.config, name)
                setattr(parent, attr_name, moe_layer)
                self.moe_layers[name.replace(".", "_")] = moe_layer
                logger.info(f"Injected MoELoRA layer: {name}")

    def forward(self, **kwargs):
        return self.base_model(**kwargs)

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []

        for layer in self.moe_layers.values():
            moe_layer = cast("MoELoRALayer", layer)
            params.extend(moe_layer.router.parameters())
            params.extend(moe_layer.lora_A_experts)
            params.extend(moe_layer.lora_B_experts)

        logger.info(f"Collected trainable MoE parameters | count={len(params)}")
        return params

    def compute_total_diversity_loss(self, method: str = "lowrank") -> Tensor:
        """Compute total diversity loss across all MoE layers.

        Args:
            method: 'lowrank' or 'subspace'
                - 'lowrank': Compares full W_i = B_i @ A_i using trace trick
                - 'subspace': Compares only input subspaces (A matrices)
                  Subspace is faster and often better theoretically.

        """
        model_params = list(self.parameters())
        if not model_params:
            return torch.tensor(0.0)

        device = model_params[0].device
        total_loss = torch.tensor(0.0, device=device)

        for layer in self.moe_layers.values():
            moe_layer = cast("MoELoRALayer", layer)
            if moe_layer.last_topk_indices is not None:
                total_loss += moe_layer.compute_diversity_loss(
                    moe_layer.last_topk_indices
                )

        logger.info(f"Total diversity loss computed using {method} method")
        return total_loss

    def print_trainable_parameters(self) -> None:
        trainable_params = 0
        all_params = 0

        for param in self.base_model.parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        percentage = 100 * trainable_params / all_params if all_params > 0 else 0.0

        logger.info(
            f"Trainable params: {trainable_params:,} | "
            f"All params: {all_params:,} | "
            f"Trainable %: {percentage:.2f}%"
        )

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        try:
            self.base_model.save_pretrained(str(save_path))

            moe_state = {
                "moe_layers": {k: v.state_dict() for k, v in self.moe_layers.items()},
                "config": self.config,
            }
            torch.save(moe_state, save_path / "moe_adapter.pt")

            logger.info(f"MoE model saved successfully at {save_path}")

        except Exception as e:
            logger.error(f"Failed to save MoE model: {e}")
            raise
