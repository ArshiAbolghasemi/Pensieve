from typing import Literal

from peft import LoraConfig


class MoLELoRAConfig(LoraConfig):
    """Configuration class for Mixture of Experts LoRA.

    Inherits from PEFT's LoraConfig and adds MoE-specific parameters.

    Args:
        r (int): Rank of LoRA adapters
        num_experts (int): Number of expert adapters
        top_k (int): Number of experts to route to per token
        lora_alpha (int): LoRA scaling parameter (default: 32)
        lora_dropout (float): Dropout probability for LoRA layers (default: 0.05)
        adapter_init (str): Initialization method for expert adapters
            Options: "random", "pissa", "goat", "milora", "goat_mini", "pissa_milora"
        router_init (str): Initialization method for router
            Options: "random", "orthogonal", "svd"
        target_modules (Union[list[str], str]): Modules to apply MoE LoRA to

    """

    def __init__(
        self,
        r: int = 16,
        num_experts: int = 8,
        top_k: int = 2,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        adapter_init: Literal[
            "random", "pissa", "goat", "milora", "goat_mini", "pissa_milora"
        ] = "random",
        router_init: Literal["random", "orthogonal", "svd"] = "random",
        target_modules: list[str] | str | None = None,
        **kwargs,
    ):
        _ = kwargs
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        super().__init__(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
        )

        self.num_experts = num_experts
        self.top_k = top_k
        self.adapter_init = adapter_init
        self.router_init = router_init

        self._validate_moe_config()

    def _validate_moe_config(self):
        """Validate MoE-specific configuration parameters."""
        if self.num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {self.num_experts}")

        if self.top_k < 1 or self.top_k > self.num_experts:
            raise ValueError(
                f"top_k must be between 1 and num_experts ({self.num_experts}), "
                f"got {self.top_k}"
            )

        if self.r < 1:
            raise ValueError(f"r (rank) must be >= 1, got {self.r}")

        valid_adapter_inits = [
            "random",
            "pissa",
            "goat",
            "milora",
            "goat_mini",
            "pissa_milora",
        ]
        if self.adapter_init not in valid_adapter_inits:
            raise ValueError(
                f"adapter_init must be one of {valid_adapter_inits}, "
                f"got {self.adapter_init}"
            )

        valid_router_inits = ["random", "orthogonal", "svd"]
        if self.router_init not in valid_router_inits:
            raise ValueError(
                f"router_init must be one of {valid_router_inits}, got {self.router_init}"
            )

    def to_dict(self) -> dict:
        """Convert config to dictionary, including MoE-specific parameters."""
        config_dict = super().to_dict()

        config_dict.update(
            {
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "adapter_init": self.adapter_init,
                "router_init": self.router_init,
            }
        )

        return config_dict

    @classmethod
    def from_dict(cls, config_dict: dict) -> "MoLELoRAConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    def __repr__(self) -> str:
        """String representation of config."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  r={self.r},\n"
            f"  num_experts={self.num_experts},\n"
            f"  top_k={self.top_k},\n"
            f"  lora_alpha={self.lora_alpha},\n"
            f"  lora_dropout={self.lora_dropout},\n"
            f"  adapter_init='{self.adapter_init}',\n"
            f"  router_init='{self.router_init}',\n"
            f"  target_modules={self.target_modules},\n"
            f"  bias='{self.bias}',\n"
            f"  task_type='{self.task_type}',\n"
            f"  peft_type={self.peft_type}\n"
            f")"
        )
