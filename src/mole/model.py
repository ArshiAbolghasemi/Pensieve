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
    """Mixture of Experts LoRA Model with PEFT compatibility.

    This model wraps a pretrained transformer and injects MoE LoRA layers
    into specified linear layers (e.g., attention projections).

    Architecture:
        Base Model (frozen)
            ↓
        MoE LoRA Layers (trainable) injected into target modules
            ↓
        Output with enhanced capacity via expert routing

    Args:
        model: Base pretrained model (typically from transformers)
        peft_config: MoLELoRAConfig with all MoE and LoRA settings
        adapter_name: PEFT adapter name (default: "default")

    """

    def __init__(
        self,
        model: PreTrainedModel,
        peft_config: MoLELoRAConfig,
        adapter_name: str = "default",
    ) -> None:
        """Initialize MoE LoRA Model.

        Args:
            model: Base pretrained model
            peft_config: MoE LoRA configuration
            adapter_name: Adapter identifier

        """
        super().__init__(model, peft_config, adapter_name)

        self.moe_config = peft_config
        self.moe_layers: nn.ModuleDict = nn.ModuleDict()

        logger.info("Starting MoE layer injection")
        self._inject_moe_layers()
        logger.info(f"MoE injection completed | layers={len(self.moe_layers)}")

    def _inject_moe_layers(self) -> None:
        """Inject MoE LoRA layers into the base model.

        This replaces target linear layers (e.g., q_proj, v_proj) with
        MoELoRALayer wrappers that add expert routing on top.

        Process:
        1. Iterate through all modules in base model
        2. Find linear layers matching target_modules
        3. Replace them with MoELoRALayer
        4. Store in self.moe_layers for easy access

        """
        injection_count = 0

        if self.moe_config.target_modules is None:
            raise ValueError("target modules is empty in moe config")

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

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Get all trainable MoE parameters (routers + expert adapters).

        Returns only the parameters that should be optimized during training:
        - Router weights (for expert selection)
        - Expert LoRA A matrices
        - Expert LoRA B matrices

        Base model parameters are NOT included (they remain frozen).

        Returns:
            List of trainable parameters

        """
        params: list[nn.Parameter] = []

        for layer in self.moe_layers.values():
            layer = cast("MoELoRALayer", layer)

            params.extend(layer.router.parameters())

            params.extend(layer.lora_A_experts)

            params.extend(layer.lora_B_experts)

        logger.info(f"Collected trainable MoE parameters | count={len(params)}")
        return params

    def compute_total_diversity_loss(self) -> Tensor:
        """Compute total diversity loss across all MoE layers.

        Diversity loss encourages experts to specialize in different aspects
        by penalizing similarity between selected experts.

        Formula:
            L_diversity = Σ_layers diversity_loss(layer)

        Where each layer's diversity loss is the average cosine similarity
        between pairs of selected experts.

        Returns:
            Scalar tensor with average diversity loss

        """
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
        """Compute Pairwise Expert Similarity (PES) metric.

        PES measures how similar expert outputs are. Lower PES indicates
        better expert specialization (experts do different things).

        Formula:
            For each layer:
                For each sample in batch:
                    similarity_matrix = cosine_similarity(expert_outputs)
                    pes_sample = mean(upper_triangle(similarity_matrix))
                pes_layer = mean(pes_sample)
            pes_model = mean(pes_layer)

        Returns:
            Dictionary with:
            - "pes_model": Average PES across all layers
            - "pes_per_layer": PES for each individual layer

        """
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

            # Compute similarity for each sample in batch
            batch_similarities = []

            for batch_idx in range(batch_size):
                sample_outputs = expert_outputs[batch_idx]  # (num_experts, out_features)

                # Normalize expert outputs
                sample_outputs_norm = F.normalize(sample_outputs, p=2, dim=1)

                # Compute similarity matrix
                similarity_matrix = torch.matmul(sample_outputs_norm, sample_outputs_norm.T)

                # Get upper triangle (avoid diagonal and duplicates)
                upper_triangle = torch.triu(similarity_matrix, diagonal=1)

                # Average similarity across expert pairs
                num_pairs = (num_experts * (num_experts - 1)) / 2
                mean_similarity = upper_triangle.sum() / num_pairs

                batch_similarities.append(mean_similarity.item())

            # Average across batch
            layer_pes = sum(batch_similarities) / len(batch_similarities)
            layer_pes_values[layer_name] = layer_pes
            logger.debug(f"Layer {layer_name} PES: {layer_pes:.4f}")

            total_pes += layer_pes
            num_layers_with_outputs += 1

        # Average across layers
        pes_model = (
            total_pes / num_layers_with_outputs if num_layers_with_outputs > 0 else 0.0
        )

        return {"pes_model": pes_model, "pes_per_layer": layer_pes_values}

    def print_trainable_parameters(self) -> None:
        """Print statistics about trainable vs total parameters.

        Useful for verifying that only MoE parameters are trainable
        and base model is frozen.

        Example output:
            Trainable params: 8,388,608 | All params: 3,820,000,000 | Trainable %: 0.22%
        """
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
        """Disable all MoE adapter layers.

        When disabled, the model acts as if only the base model exists
        (no expert routing or LoRA adaptations).

        Useful for:
        - Comparing performance with/without adapters
        - Inference with base model only
        """
        for layer in self.moe_layers.values():
            layer = cast("MoELoRALayer", layer)
            layer.disable_adapters = True
        logger.info("Disabled all MoE adapter layers")

    def enable_adapter_layers(self) -> None:
        """Enable all MoE adapter layers.

        Re-enables adapters after they've been disabled.
        """
        for layer in self.moe_layers.values():
            layer = cast("MoELoRALayer", layer)
            layer.disable_adapters = False
        logger.info("Enabled all MoE adapter layers")

    def set_adapter(self, adapter_name: str) -> None:
        """Set active adapter (for PEFT compatibility).

        Currently only supports the "default" adapter.

        Args:
            adapter_name: Name of adapter to activate

        """
        if adapter_name != "default":
            logger.warning(
                f"Only 'default' adapter is supported, got {adapter_name}. "
                f"Multi-adapter support coming soon."
            )

    def merge_and_unload(self) -> PreTrainedModel:
        """Merge adapter weights into base model and return base model.

        WARNING: Not yet implemented for MoE adapters due to complexity
        of merging multiple expert adaptations.

        Returns:
            Base model (without merging for now)

        """
        logger.warning(
            "Merging MoE adapters not yet implemented. "
            "Returning base model without merging."
        )
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
        """Load a pretrained MoE LoRA model from checkpoint.

        This method signature matches PeftModel.from_pretrained for compatibility
        with the PEFT ecosystem.

        Args:
            model: Base pretrained model
            model_id: Path to checkpoint directory
            adapter_name: Name of adapter to load (default: "default")
            is_trainable: Whether to load in training mode (default: False)
            config: Optional config override (default: load from checkpoint)
            autocast_adapter_dtype: PEFT compatibility parameter
            ephemeral_gpu_offload: PEFT compatibility parameter
            low_cpu_mem_usage: PEFT compatibility parameter
            key_mapping: PEFT compatibility parameter
            **kwargs: Additional arguments

        Returns:
            MoELoRAModel with loaded weights

        Raises:
            FileNotFoundError: If moe_adapter.pt not found
            KeyError: If config not found in checkpoint
        """
        # Suppress unused parameter warnings (PEFT compatibility)
        _ = (
            autocast_adapter_dtype,
            ephemeral_gpu_offload,
            low_cpu_mem_usage,
            key_mapping,
            kwargs,
        )

        # Convert PathLike to string
        model_id = str(model_id)

        # Load checkpoint
        moe_state_path = os.path.join(model_id, "moe_adapter.pt")
        if not os.path.exists(moe_state_path):
            raise FileNotFoundError(
                f"MoE adapter not found at {moe_state_path}. "
                f"Make sure the checkpoint was saved with save_pretrained()."
            )

        logger.info(f"Loading MoE adapter from {moe_state_path}")
        moe_state = torch.load(moe_state_path, map_location="cpu")

        # Get configuration
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
                f"Provided config type {type(config)} is not MoLELoRAConfig. "
                f"Attempting to load from checkpoint instead."
            )
            if "config" not in moe_state:
                raise KeyError(
                    "Config not found in checkpoint and provided config is invalid"
                )
            moe_config = moe_state["config"]

        logger.info(
            f"Creating MoE model with {len(moe_state.get('moe_layers', {}))} saved layers"
        )

        # Create model instance
        moe_model = cls(
            model=cast("PreTrainedModel", model),
            peft_config=moe_config,
            adapter_name=adapter_name,
        )

        # Load layer weights
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
                        f"Layer {layer_name} from checkpoint not found in model. "
                        f"This may happen if target_modules changed."
                    )

            logger.info(
                f"Layer loading complete: {loaded_count} loaded, {failed_count} failed"
            )
        else:
            logger.warning(
                "No 'moe_layers' found in checkpoint - model will use random initialization"
            )

        # Set training mode
        if not is_trainable:
            moe_model.eval()
            for param in moe_model.parameters():
                param.requires_grad = False
            logger.info("Model loaded in evaluation mode (frozen)")
        else:
            moe_model.train()
            # Enable gradients only for MoE parameters
            for param in moe_model.base_model.parameters():
                param.requires_grad = False
            for param in moe_model.get_trainable_parameters():
                param.requires_grad = True
            logger.info("Model loaded in training mode (MoE parameters trainable)")

        logger.info(f"Successfully loaded MoE model from {model_id}")
        return moe_model
