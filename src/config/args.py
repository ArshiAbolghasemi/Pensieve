import argparse


def get_train_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MoE LoRA model")

    parser.add_argument(
        "--model_name", type=str, required=True, help="HuggingFace model identifier"
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=True,
        help="Load model in 4-bit quantization",
    )
    parser.add_argument(
        "--use_flash_attention", action="store_true", help="Use flash attention"
    )

    parser.add_argument("--num_experts", type=int, default=4, help="Number of experts")
    parser.add_argument(
        "--top_k", type=int, default=2, help="Number of experts to activate per token"
    )
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument(
        "--lora_alpha", type=int, default=32, help="LoRA alpha (scaling factor)"
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05, help="LoRA dropout rate"
    )
    parser.add_argument(
        "--adapter_init",
        type=str,
        default="pissa",
        choices=["random", "pissa", "milora", "goat", "goat_mini", "pissa_milora"],
        help="Expert adapter initialization method",
    )
    parser.add_argument(
        "--router_init",
        type=str,
        default="svd",
        choices=["random", "orthogonal", "svd"],
        help="Router initialization method",
    )
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        help="Target modules to apply MoE LoRA",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--samples_per_task",
        type=int,
        default=500,
        help="Number of samples per task from FLAN dataset",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps for learning rate scheduler",
    )
    parser.add_argument(
        "--diversity_loss_coefficient",
        type=float,
        default=0.01,
        help="coefficient for diversity loss",
    )

    return parser


def get_benchmark_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate models on benchmarks")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Base model name or path",
    )
    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        default="checkpoints/single_lora",
        help="Path to LoRA checkpoint",
    )
    parser.add_argument(
        "--moe_checkpoint",
        type=str,
        required=True,
        help="Path to MoE checkpoint",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run evaluation on",
    )
    parser.add_argument(
        "--skip_base",
        action="store_true",
        default=False,
        help="Skip base model evaluation",
    )
    parser.add_argument(
        "--skip_lora",
        action="store_true",
        default=False,
        help="Skip single LoRA model evaluation",
    )
    parser.add_argument(
        "--skip_moe",
        action="store_true",
        default=False,
        help="Skip MoE model evaluation",
    )
    return parser
