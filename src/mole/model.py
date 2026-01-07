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
    """MoE LoRA Model with proper PeftModel integration."""

    def __init__(
        self,
        model: PreTrainedModel,
        peft_config: MoLELoRAConfig,
        adapter_name: str = "default",
    ) -> None:
        """Initialize MoE LoRA Model.

        CRITICAL: We bypass PeftModel.__init__() to prevent automatic
        adapter creation that would conflict with our MoE layers.

        Args:
            model: Base pretrained model
            peft_config: MoE LoRA configuration
            adapter_name: Adapter identifier

        """
        super().__init__(model, peft_config, adapter_name)

        nn.Module.__init__(self)

        self.base_model = model
        self.config = {}
        self.active_adapter = adapter_name
        self.peft_config = {adapter_name: peft_config}
        self.peft_type = peft_config.peft_type

        self.moe_config = peft_config
        self.moe_layers: nn.ModuleDict = nn.ModuleDict()

        logger.info("Starting MoE layer injection")
        self._inject_moe_layers()
        logger.info(f"MoE injection completed | layers={len(self.moe_layers)}")

    def add_adapter(
        self, adapter_name: str, peft_config: PeftConfig, low_cpu_mem_usage: bool = False
    ) -> None:
        """Override add_adapter to prevent PEFT from creating standard adapters.

        CRITICAL: This method is called by PeftModel.__init__() which we skip,
        but we override it anyway for safety in case it's called elsewhere.

        Our MoE layers are created in _inject_moe_layers() instead.
        """
        _ = peft_config, low_cpu_mem_usage

        logger.info(
            f"add_adapter called for '{adapter_name}', but MoE model handles "
            "adapter creation via _inject_moe_layers(). Ignoring PEFT's auto-creation."
        )
        # Do nothing - we create our own adapters in _inject_moe_layers()

    def _inject_moe_layers(self) -> None:
        """Inject MoE LoRA layers into the base model.

        This is our custom adapter creation logic that replaces
        PEFT's standard adapter creation.
        """
        injection_count = 0

        if self.moe_config.target_modules is None:
            raise ValueError("target_modules is empty in moe config")

        for name, module in self.base_model.named_modules():
            should_inject = any(
                target in name for target in self.moe_config.target_modules
            ) and isinstance(module, nn.Linear)

            if not should_inject:
                continue

            parent_name = ".".join(name.split(".")[:-1])
            attr_name = name.split(".")[-1]
            parent = (
                self.base_model.get_submodule(parent_name)
                if parent_name
                else self.base_model
            )

            moe_layer = MoELoRALayer(
                base_layer=cast("nn.Linear", module),
                config=self.moe_config,
                layer_name=name,
            )

            setattr(parent, attr_name, moe_layer)

            safe_name = name.replace(".", "_")
            self.moe_layers[safe_name] = moe_layer

            injection_count += 1
            logger.info(f"Injected MoE layer [{injection_count}]: {name}")

    def get_base_model(self) -> nn.Module:
        """Get the base model (required by PeftModel).

        This is called by PeftModel.forward() to route inputs.
        """
        return self.base_model

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
                    "Call forward() first to compute PES."
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
            logger.debug(f"Layer {layer_name} PES: {layer_pes:.4f}")

            total_pes += layer_pes
            num_layers_with_outputs += 1

        pes_model = (
            total_pes / num_layers_with_outputs if num_layers_with_outputs > 0 else 0.0
        )

        return {"pes_model": pes_model, "pes_per_layer": layer_pes_values}

    def print_trainable_parameters(self) -> None:
        """Print statistics about trainable vs total parameters."""
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
        logger.info("Disabled all MoE adapter layers")

    def enable_adapter_layers(self) -> None:
        """Enable all MoE adapter layers."""
        for layer in self.moe_layers.values():
            layer = cast("MoELoRALayer", layer)
            layer.disable_adapters = False
        logger.info("Enabled all MoE adapter layers")

    def set_adapter(self, adapter_name: str) -> None:
        """Set active adapter (for PEFT compatibility).

        Note: Currently only supports single adapter.
        """
        if adapter_name != self.active_adapter:
            logger.warning(
                f"Only '{self.active_adapter}' adapter is supported, got {adapter_name}. "
                f"Multi-adapter support coming soon."
            )
        self.active_adapter = adapter_name

    def merge_and_unload(self) -> PreTrainedModel:
        """Merge adapter weights into base model.

        WARNING: Not yet implemented for MoE adapters.
        """
        logger.warning(
            "Merging MoE adapters not yet implemented. "
            "Returning base model without merging."
        )
        return cast("PreTrainedModel", self.base_model)

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = True,
        selected_adapters: list[str] | None = None,
        save_embedding_layers: str | bool = "auto",
        is_main_process: bool = True,
        path_initial_model_for_weight_conversion: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Save MoE adapters to directory.

        This saves:
        - MoE configuration
        - All MoE layer weights (routers + experts)
        """
        _ = (
            safe_serialization,
            selected_adapters,
            save_embedding_layers,
            is_main_process,
            path_initial_model_for_weight_conversion,
            kwargs,
        )
        os.makedirs(save_directory, exist_ok=True)

        moe_state = {
            "config": self.moe_config,
            "moe_layers": {
                name: layer.state_dict() for name, layer in self.moe_layers.items()
            },
        }

        save_path = os.path.join(save_directory, "moe_adapter.pt")
        torch.save(moe_state, save_path)
        logger.info(f"Saved MoE adapter to {save_path}")

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
        """Load a pretrained MoE LoRA model from checkpoint.

        Args:
            model: Base pretrained model
            model_id: Path to checkpoint directory
            adapter_name: Name of adapter to load
            is_trainable: Whether to load in training mode
            config: Optional config override
            **kwargs: Additional arguments (for compatibility)

        Returns:
            MoELoRAModel with loaded weights

        """
        _ = (
            autocast_adapter_dtype,
            ephemeral_gpu_offload,
            low_cpu_mem_usage,
            key_mapping,
            kwargs,
        )
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
                    f"The checkpoint may be corrupted."
                )
            moe_config = moe_state["config"]
        elif isinstance(config, MoLELoRAConfig):
            moe_config = config
        else:
            logger.warning(
                f"Provided config type {type(config)} is not MoLELoRAConfig. "
                f"Loading from checkpoint instead."
            )
            moe_config = moe_state.get("config")

        logger.info(
            f"Creating MoE model with {len(moe_state.get('moe_layers', {}))} saved layers"
        )

        moe_model = cls(
            model=cast("PreTrainedModel", model),
            peft_config=moe_config,
            adapter_name=adapter_name,
        )

        if "moe_layers" in moe_state:
            logger.info(f"Loading {len(moe_state['moe_layers'])} MoE layer states")
            loaded_count = 0
            failed_count = 0

            for layer_name, layer_state in moe_state["moe_layers"].items():
                if layer_name in moe_model.moe_layers:
                    try:
                        moe_model.moe_layers[layer_name].load_state_dict(
                            layer_state, strict=False
                        )
                        loaded_count += 1
                        logger.debug(f"Loaded layer: {layer_name}")
                    except Exception as e:
                        failed_count += 1
                        logger.warning(f"Failed to load layer {layer_name}: {e}")
                else:
                    failed_count += 1
                    logger.warning(
                        f"Layer {layer_name} from checkpoint not found in model."
                    )

            logger.info(
                f"Layer loading complete: {loaded_count} loaded, {failed_count} failed"
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
            for param in moe_model.base_model.parameters():
                param.requires_grad = False
            for param in moe_model.get_trainable_parameters():
                param.requires_grad = True
            logger.info("Model loaded in training mode (MoE parameters trainable)")

        logger.info(f"Successfully loaded MoE model from {model_id}")
        return moe_model
