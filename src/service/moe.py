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
logger.setLevel(logging.INFO)


class MoELoRALayer(nn.Module):
    """Mixture of Experts LoRA Layer with configurable initialization."""

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

        logger.info(f"[{self.layer_name}] Initializing router...")
        self.router = self._initialize_router()
        logger.info(f"[{self.layer_name}] Router initialized.")

        logger.info(f"[{self.layer_name}] Initializing LoRA experts...")
        self.lora_A_experts, self.lora_B_experts = self._initialize_experts()
        logger.info(f"[{self.layer_name}] LoRA experts initialized.")

        self.dropout = (
            nn.Dropout(p=self.lora_dropout) if self.lora_dropout > 0 else nn.Identity()
        )

        self.residual_weight = None
        logger.info(f"[{self.layer_name}] MoELoRALayer initialization complete.")

    def _initialize_router(self) -> nn.Linear:
        router = nn.Linear(self.in_features, self.num_experts, bias=False)
        logger.info(f"[{self.layer_name}] Router init method: {self.config.router_init}")

        if self.config.router_init == "random":
            nn.init.normal_(router.weight, mean=0.0, std=0.01)
        elif self.config.router_init == "orthogonal":
            with torch.no_grad():
                nn.init.orthogonal_(router.weight)
        elif self.config.router_init == "svd":
            R = svd_router_initialization(
                weights=self.base_layer,
                num_experts=self.num_experts,
                in_dim=self.in_features,
            )
            with torch.no_grad():
                router.weight.copy_(R.to(router.weight.dtype))
        else:
            raise ValueError(f"Unknown router_init: {self.config.router_init}")

        logger.info(f"[{self.layer_name}] Router weights shape: {router.weight.shape}")
        return router

    def _initialize_experts(self) -> tuple[nn.ParameterList, nn.ParameterList]:
        logger.info(f"[{self.layer_name}] Adapter init method: {self.config.adapter_init}")
        if self.config.adapter_init == "random":
            return self._initialize_experts_random()
        return self._initialize_experts_svd()

    def _initialize_experts_random(self) -> tuple[nn.ParameterList, nn.ParameterList]:
        lora_A_list = nn.ParameterList()
        lora_B_list = nn.ParameterList()
        for i in range(self.num_experts):
            lora_A = nn.Parameter(torch.empty(self.r, self.in_features))
            nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
            lora_B = nn.Parameter(torch.zeros(self.out_features, self.r))
            lora_A_list.append(lora_A)
            lora_B_list.append(lora_B)
            logger.debug(f"[{self.layer_name}] Expert {i} initialized (random).")
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

        logger.info(f"[{self.layer_name}] Initializing experts with SVD method: {method}")
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
            self.residual_weight = residual_weight

        lora_A_params = nn.ParameterList([nn.Parameter(a) for a in lora_A_list])
        lora_B_params = nn.ParameterList([nn.Parameter(b) for b in lora_B_list])
        logger.info(f"[{self.layer_name}] SVD experts initialized.")
        return lora_A_params, lora_B_params

    def compute_orthogonal_loss(self) -> Tensor:
        R = self.router.weight
        RRT = R @ R.T
        I = torch.eye(self.num_experts, device=R.device, dtype=R.dtype)
        loss = torch.norm(RRT - I, p="fro") ** 2
        logger.debug(f"[{self.layer_name}] Orthogonal loss computed: {loss.item():.6f}")
        return loss

    def forward(
        self, x: Tensor, return_aux_loss: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        original_shape = x.shape
        x_flat = x.view(-1, self.in_features)

        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

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

        if self.config.adapter_init == "random":
            final_output = base_output + expert_output
        else:
            if self.residual_weight is not None:
                residual_output = F.linear(x, self.residual_weight, None)
                final_output = base_output - residual_output + expert_output
            else:
                final_output = base_output + expert_output

        if return_aux_loss:
            aux_loss = self.compute_orthogonal_loss()
            return final_output, aux_loss

        return final_output


class MoELoRAModel:
    def __init__(self, model: PreTrainedModel, config: MoELoRAConfig) -> None:
        self.model = model
        self.config = config
        self.moe_layers = {}
        logger.info("Injecting MoE layers into model...")
        self._inject_moe_layers()
        logger.info(f"{len(self.moe_layers)} MoE layers injected.")

    def _inject_moe_layers(self) -> None:
        for name, module in self.model.named_modules():
            if any(t in name for t in self.config.target_modules) and isinstance(
                module, nn.Linear
            ):
                parent_name = ".".join(name.split(".")[:-1])
                attr_name = name.split(".")[-1]
                parent = (
                    self.model.get_submodule(parent_name) if parent_name else self.model
                )

                moe_layer = MoELoRALayer(module, self.config, name)
                setattr(parent, attr_name, moe_layer)
                self.moe_layers[name] = moe_layer
                logger.info(f"Injected MoE layer: {name}")

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        params = []
        for name, layer in self.moe_layers.items():
            params.extend(layer.router.parameters())
            params.extend(layer.lora_A_experts.parameters())
            params.extend(layer.lora_B_experts.parameters())
            logger.debug(
                f"[{name}] Added {len(list(layer.parameters()))} trainable parameters."
            )
        return params

    def compute_total_orthogonal_loss(self) -> Tensor:
        total_loss = torch.tensor(0.0, device=next(iter(self.model.parameters())).device)
        for name, layer in self.moe_layers.items():
            loss = layer.compute_orthogonal_loss()
            total_loss += loss
            logger.debug(f"[{name}] Orthogonal loss: {loss.item():.6f}")
        logger.info(f"Total orthogonal loss across all MoE layers: {total_loss.item():.6f}")
        return total_loss

    def print_trainable_parameters(self) -> None:
        trainable_params = 0
        all_params = 0
        for param in self.model.parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        logger.info(
            f"Trainable params: {trainable_params:,} | "
            f"All params: {all_params:,} | "
            f"Trainable %: {100 * trainable_params / all_params:.2f}%"
        )
