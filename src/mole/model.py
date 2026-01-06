import logging
import os
from typing import Any, cast

import torch
import torch.nn.functional as F
from peft import PeftConfig, PeftModel
from torch import Tensor, nn
from transformers import PreTrainedModel

from config.mole import MoLELoRAConfig
from mole.layer import MoELoRALayer

logger = logging.getLogger(__name__)


class MoELoRAModel(PeftModel):
    """PEFT-compatible MoE LoRA Model."""

    def __init__(
        self,
        model: PreTrainedModel,
        peft_config: MoLELoRAConfig,
        adapter_name: str = "default",
    ) -> None:
        super().__init__(model, cast("PeftConfig", {}), adapter_name)

        self.moe_config = peft_config
        self.moe_layers: nn.ModuleDict = nn.ModuleDict()

        logger.info("Starting MoE layer injection")
        self._inject_moe_layers()
        logger.info(f"MoE injection completed | layers={len(self.moe_layers)}")

    def _inject_moe_layers(self) -> None:
        """Inject MoE LoRA layers into the base model."""
        for name, module in self.base_model.named_modules():
            if any(t in name for t in self.moe_config.target_modules) and isinstance(
                module, nn.Linear
            ):
                parent_name = ".".join(name.split(".")[:-1])
                attr_name = name.split(".")[-1]
                parent = (
                    self.base_model.get_submodule(parent_name)
                    if parent_name
                    else self.base_model
                )

                moe_layer = MoELoRALayer(
                    module, self.moe_config, name, adapter_name="default"
                )
                setattr(parent, attr_name, moe_layer)
                self.moe_layers[name.replace(".", "_")] = moe_layer
                logger.info(f"Injected MoELoRA layer: {name}")

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Get all trainable MoE parameters."""
        params: list[nn.Parameter] = []

        for layer in self.moe_layers.values():
            layer = cast("MoELoRALayer", layer)
            params.extend(layer.router.parameters())
            params.extend(layer.lora_A_experts)
            params.extend(layer.lora_B_experts)

        logger.info(f"Collected trainable MoE parameters | count={len(params)}")
        return params

    def compute_total_diversity_loss(self) -> Tensor:
        """Compute total diversity loss across all MoE layers."""
        model_params = list(self.parameters())
        if not model_params:
            return torch.tensor(0.0)

        device = model_params[0].device
        total_loss = torch.tensor(0.0, device=device)

        for layer in self.moe_layers.values():
            layer = cast("MoELoRALayer", layer)
            if layer.last_topk_indices is not None:
                total_loss += layer.compute_diversity_loss(layer.last_topk_indices)

        return total_loss

    def compute_pairwise_expert_similarity(self) -> dict:
        """Compute Pairwise Expert Similarity (PES) metric."""
        layer_pes_values = {}
        total_pes = 0.0
        num_layers_with_outputs = 0

        for layer_name, layer in self.moe_layers.items():
            layer = cast("MoELoRALayer", layer)
            if layer.last_expert_outputs is None:
                logger.warning(
                    f"Layer {layer_name} has no stored expert outputs. "
                    "Call forward() first."
                )
                continue

            expert_outputs = layer.last_expert_outputs
            batch_size = expert_outputs.shape[0]
            num_experts = expert_outputs.shape[1]

            if num_experts < 2:
                layer_pes_values[layer_name] = 0.0
                continue

            batch_similarities = []

            for batch_idx in range(batch_size):
                sample_outputs = expert_outputs[batch_idx]
                sample_outputs_norm = F.normalize(sample_outputs, p=2, dim=1)
                similarity_matrix = torch.matmul(sample_outputs_norm, sample_outputs_norm.T)
                upper_triangle = torch.triu(similarity_matrix, diagonal=1)
                num_pairs = (num_experts * (num_experts - 1)) / 2
                mean_similarity = upper_triangle.sum() / num_pairs
                batch_similarities.append(mean_similarity.item())

            layer_pes = sum(batch_similarities) / len(batch_similarities)
            layer_pes_values[layer_name] = layer_pes
            logger.info(f"layer {layer_name} pes: {layer_pes}")

            total_pes += layer_pes
            num_layers_with_outputs += 1

        pes_model = (
            total_pes / num_layers_with_outputs if num_layers_with_outputs > 0 else 0.0
        )

        return {"pes_model": pes_model, "pes_per_layer": layer_pes_values}

    def print_trainable_parameters(self) -> None:
        """Print trainable parameters statistics."""
        trainable_params = 0
        all_params = 0

        for param in self.parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        percentage = 100 * trainable_params / all_params if all_params > 0 else 0.0

        logger.info(
            f"Trainable params: {trainable_params:,} | "
            f"All params: {all_params:,} | "
            f"Trainable %: {percentage:.2f}%"
        )

    def disable_adapter_layers(self) -> None:
        """Disable all MoE adapter layers."""
        for layer in self.moe_layers.values():
            layer = cast("MoELoRALayer", layer)
            layer.disable_adapters = True

    def enable_adapter_layers(self) -> None:
        """Enable all MoE adapter layers."""
        for layer in self.moe_layers.values():
            layer = cast("MoELoRALayer", layer)
            layer.disable_adapters = False

    def set_adapter(self, adapter_name: str) -> None:
        """Set active adapter (for PEFT compatibility)."""
        if adapter_name != "default":
            logger.warning(f"Only 'default' adapter is supported, got {adapter_name}")

    def merge_and_unload(self) -> PreTrainedModel:
        """Merge adapter weights into base model and return base model."""
        logger.warning("Merging MoE adapters not yet implemented")
        return cast("PreTrainedModel", self.base_model)

    @classmethod
    def from_pretrained(
        cls,
        model: torch.nn.Module,
        model_id: str | os.PathLike,
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: PeftConfig | None = None,
        autocast_adapter_dtype: bool = True,
        ephemeral_gpu_offload: bool = False,
        low_cpu_mem_usage: bool = False,
        key_mapping: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> "MoELoRAModel":
        _ = (
            autocast_adapter_dtype,
            ephemeral_gpu_offload,
            low_cpu_mem_usage,
            key_mapping,
            kwargs,
        )
        """Load a pretrained MoE LoRA model.

        This method signature matches PeftModel.from_pretrained for compatibility.

        Args:
            model: Base model (typically PreTrainedModel)
            model_id: Path to the checkpoint directory
            adapter_name: Name of the adapter (default: "default")
            is_trainable: Whether model should be trainable (default: False)
            config: Optional PeftConfig (will load from checkpoint if None)
            autocast_adapter_dtype: Whether to autocast adapter dtype (for compatibility)
            ephemeral_gpu_offload: Whether to use ephemeral GPU offload (for compatibility)
            low_cpu_mem_usage: Whether to use low CPU memory (for compatibility)
            key_mapping: Optional key mapping for state dict (for compatibility)
            **kwargs: Additional arguments

        Returns:
            MoELoRAModel instance with loaded weights

        Raises:
            FileNotFoundError: If checkpoint file not found
            KeyError: If config not found in checkpoint

        Example:
            >>> from transformers import AutoModelForCausalLM
            >>> base_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
            >>> moe_model = MoELoRAModel.from_pretrained(
            ...     model=base_model,
            ...     model_id="path/to/checkpoint",
            ...     adapter_name="default"
            ... )

        """
        model_id = str(model_id)

        moe_state_path = os.path.join(model_id, "moe_adapter.pt")
        if not os.path.exists(moe_state_path):
            raise FileNotFoundError(
                f"MoE adapter not found at {moe_state_path}. "
                f"Make sure the checkpoint was saved with save_pretrained()."
            )

        logger.info(f"Loading MoE adapter from {moe_state_path}")
        moe_state = torch.load(moe_state_path, map_location="cpu")

        if config is None:
            if "config" not in moe_state:
                raise KeyError(
                    f"Config not found in checkpoint at {moe_state_path}. "
                    f"The checkpoint may be corrupted or from an older version."
                )
            moe_config = moe_state["config"]
        elif isinstance(config, MoLELoRAConfig):
            moe_config = config
        else:
            logger.warning(
                "Provided config is not MoLELoRAConfig, attempting to load from checkpoint"
            )
            if "config" not in moe_state:
                raise KeyError(
                    "Config not found in checkpoint and provided config is invalid"
                )
            moe_config = moe_state["config"]

        logger.info(
            f"Creating MoE model with {len(moe_state.get('moe_layers', {}))} layers"
        )

        moe_model = cls(
            model=cast("PreTrainedModel", model),
            peft_config=moe_config,
            adapter_name=adapter_name,
        )

        if "moe_layers" in moe_state:
            logger.info(f"Loading {len(moe_state['moe_layers'])} MoE layer states")
            for layer_name, layer_state in moe_state["moe_layers"].items():
                if layer_name in moe_model.moe_layers:
                    try:
                        moe_model.moe_layers[layer_name].load_state_dict(
                            layer_state, strict=False
                        )
                        logger.debug(f"Loaded layer: {layer_name}")
                    except Exception as e:
                        logger.warning(f"Failed to load layer {layer_name}: {e}")
                else:
                    logger.warning(
                        f"Layer {layer_name} from checkpoint not found in model. "
                        f"This may happen if target_modules changed."
                    )
        else:
            logger.warning("No 'moe_layers' found in checkpoint")

        if not is_trainable:
            moe_model.eval()
            for param in moe_model.parameters():
                param.requires_grad = False
            logger.info("Model loaded in evaluation mode (frozen)")
        else:
            moe_model.train()
            logger.info("Model loaded in training mode")

        logger.info(f"Successfully loaded MoE model from {model_id}")
        return moe_model
