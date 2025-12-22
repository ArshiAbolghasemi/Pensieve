import logging
import math

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

    base_layer: nn.Linear
    config: MoELoRAConfig
    layer_name: str

    in_features: int
    out_features: int
    r: int
    num_experts: int
    top_k: int
    lora_alpha: float
    lora_dropout: float
    scaling: float

    router: nn.Linear
    lora_A_experts: nn.ParameterList
    lora_B_experts: nn.ParameterList
    dropout: nn.Module
    residual_weight: Tensor | None

    def __init__(
        self,
        base_layer: nn.Linear,
        config: MoELoRAConfig,
        layer_name: str,
    ) -> None:
        super().__init__()

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

    def _initialize_router(self) -> nn.Linear:
        router: nn.Linear = nn.Linear(self.in_features, self.num_experts, bias=False)

        if self.config.router_init == "random":
            nn.init.normal_(router.weight, mean=0.0, std=0.01)

        elif self.config.router_init == "orthogonal":
            with torch.no_grad():
                nn.init.orthogonal_(router.weight)

        elif self.config.router_init == "svd":
            R: Tensor = svd_router_initialization(
                weights=self.base_layer,
                num_experts=self.num_experts,
                in_dim=self.in_features,
            )
            with torch.no_grad():
                router.weight.copy_(R.to(router.weight.dtype))

        else:
            raise ValueError(f"Unknown router_init: {self.config.router_init}")

        return router

    def _initialize_experts(
        self,
    ) -> tuple[nn.ParameterList, nn.ParameterList]:
        if self.config.adapter_init == "random":
            return self._initialize_experts_random()
        return self._initialize_experts_svd()

    def _initialize_experts_random(
        self,
    ) -> tuple[nn.ParameterList, nn.ParameterList]:
        lora_A_list: nn.ParameterList = nn.ParameterList()
        lora_B_list: nn.ParameterList = nn.ParameterList()

        for _ in range(self.num_experts):
            lora_A = nn.Parameter(torch.empty(self.r, self.in_features))
            nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))

            lora_B = nn.Parameter(torch.zeros(self.out_features, self.r))

            lora_A_list.append(lora_A)
            lora_B_list.append(lora_B)

        return lora_A_list, lora_B_list

    def _initialize_experts_svd(
        self,
    ) -> tuple[nn.ParameterList, nn.ParameterList]:
        method_map: dict[str, DecompositionMethod] = {
            "pissa": DecompositionMethod.PISSA,
            "milora": DecompositionMethod.MILORA,
            "goat": DecompositionMethod.GOAT,
            "goat_mini": DecompositionMethod.GOAT_MINI,
            "pissa_milora": DecompositionMethod.PISSA_MILORA,
        }

        method: DecompositionMethod = method_map[self.config.adapter_init]

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

            (
                lora_A_list,
                lora_B_list,
                residual_weight,
            ) = initializer.execute()

            self.residual_weight = residual_weight

        lora_A_params = nn.ParameterList([nn.Parameter(a) for a in lora_A_list])
        lora_B_params = nn.ParameterList([nn.Parameter(b) for b in lora_B_list])

        return lora_A_params, lora_B_params

    def compute_orthogonal_loss(self) -> Tensor:
        """
        Compute orthogonal regularization loss for router: ||RR^T - I||_F^2

        This encourages the router weight matrix to have orthogonal rows,
        which helps prevent expert collapse and promotes diverse routing.

        Returns:
            Orthogonal loss scalar
        """
        R = self.router.weight

        RRT = R @ R.T

        I = torch.eye(self.num_experts, device=R.device, dtype=R.dtype)

        return torch.norm(RRT - I, p="fro") ** 2

    def forward(
        self, x: Tensor, return_aux_loss: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Forward pass with expert routing.

        Args:
            x: Input tensor of shape (batch_size, seq_len, in_features)
            return_aux_loss: If True, return (output, aux_loss) tuple

        Returns:
            Output tensor or (output, aux_loss) tuple
        """
        original_shape: torch.Size = x.shape
        x_flat: Tensor = x.view(-1, self.in_features)

        router_logits: Tensor = self.router(x_flat)
        router_probs: Tensor = F.softmax(router_logits, dim=-1)

        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)

        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        batch_seq_size: int = x_flat.shape[0]
        expert_output: Tensor = torch.zeros(
            batch_seq_size,
            self.out_features,
            device=x.device,
            dtype=x.dtype,
        )

        for k_idx in range(self.top_k):
            expert_idx: Tensor = topk_indices[:, k_idx]
            expert_weight: Tensor = topk_probs[:, k_idx : k_idx + 1]

            for expert_id in range(self.num_experts):
                mask: Tensor = expert_idx == expert_id
                if not mask.any():
                    continue

                x_expert: Tensor = x_flat[mask]
                weight_expert: Tensor = expert_weight[mask]

                lora_A: Tensor = self.lora_A_experts[expert_id]
                lora_B: Tensor = self.lora_B_experts[expert_id]

                x_dropped: Tensor = self.dropout(x_expert)
                lora_out: Tensor = (x_dropped @ lora_A.T) @ lora_B.T

                lora_out = lora_out * self.scaling

                expert_output[mask] += weight_expert * lora_out

        expert_output = expert_output.view(*original_shape[:-1], self.out_features)

        if self.config.adapter_init == "random":
            base_output: Tensor = self.base_layer(x)
            final_output = base_output + expert_output
        else:
            base_output: Tensor = self.base_layer(x)

            if self.residual_weight is not None:
                residual_output: Tensor = F.linear(x, self.residual_weight, None)
                final_output = base_output - residual_output + expert_output
            else:
                final_output = base_output + expert_output

        if return_aux_loss:
            aux_loss = self.compute_orthogonal_loss()
            return final_output, aux_loss

        return final_output


class MoELoRAModel:
    model: PreTrainedModel
    config: MoELoRAConfig
    moe_layers: dict[str, MoELoRALayer]

    def __init__(self, model: PreTrainedModel, config: MoELoRAConfig) -> None:
        self.model = model
        self.config = config
        self.moe_layers = {}

        self._inject_moe_layers()

    def _inject_moe_layers(self) -> None:
        for name, module in self.model.named_modules():
            if any(t in name for t in self.config.target_modules) and isinstance(
                module, nn.Linear
            ):
                parent_name: str = ".".join(name.split(".")[:-1])
                attr_name: str = name.split(".")[-1]

                parent = (
                    self.model.get_submodule(parent_name) if parent_name else self.model
                )

                moe_layer = MoELoRALayer(module, self.config, name)

                setattr(parent, attr_name, moe_layer)
                self.moe_layers[name] = moe_layer

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for layer in self.moe_layers.values():
            params.extend(layer.router.parameters())
            params.extend(layer.lora_A_experts.parameters())
            params.extend(layer.lora_B_experts.parameters())
        return params

    def compute_total_orthogonal_loss(self) -> Tensor:
        """
        Compute total orthogonal loss across all MoE layers.

        Returns:
            Total orthogonal loss
        """
        total_loss = torch.tensor(0.0, device=next(iter(self.model.parameters())).device)
        for layer in self.moe_layers.values():
            total_loss += layer.compute_orthogonal_loss()
        return total_loss

    def print_trainable_parameters(self) -> None:
        trainable_params: int = 0
        all_params: int = 0

        for param in self.model.parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        logger.info(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_params:,} || "
            f"Trainable %: {100 * trainable_params / all_params:.2f}"
        )
