from dataclasses import dataclass, field
from typing import Literal


@dataclass
class MoELoRAConfig:
    r: int
    num_experts: int
    top_k: int
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    adapter_init: Literal[
        "random", "pissa", "goat", "milora", "goat_mini", "pissa_milora"
    ] = "random"
    router_init: Literal["random", "orthogonal", "svd"] = "random"
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"]
    )
