import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import PreTrainedModel

from config.args import get_benchmark_arg_parser
from mole.model import MoELoRAModel
from service.model import get_model, get_tokenizer
from transfer_learning.evaluate import BenchmarkEvaluator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_base_model(model_name: str, device: str) -> PreTrainedModel:
    """Load base model without any adapters."""
    logger.info(f"Loading base model: {model_name}")
    model = get_model(
        model_name=model_name,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        use_flash_attention=False,
        device_map=device,
    )

    # Disable cache to avoid DynamicCache compatibility issues
    if hasattr(model, "config"):
        model.config.use_cache = False

    return model


def load_lora_model(
    model_name: str,
    checkpoint_path: str,
    device: str,
) -> PeftModel:
    """Load LoRA model from checkpoint."""
    logger.info(f"Loading LoRA model from: {checkpoint_path}")
    base_model = get_model(
        model_name=model_name,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        use_flash_attention=False,
    )

    # Disable cache to avoid DynamicCache compatibility issues
    if hasattr(base_model, "config"):
        base_model.config.use_cache = False

    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    return model.to(device)


def load_moe_model(
    model_name: str,
    checkpoint_path: str,
    device: str,
) -> MoELoRAModel:
    """Load MoE model from checkpoint."""
    logger.info(f"Loading MoE model from: {checkpoint_path}")

    base_model = get_model(
        model_name=model_name,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        use_flash_attention=False,
    )

    # Disable cache to avoid DynamicCache compatibility issues
    if hasattr(base_model, "config"):
        base_model.config.use_cache = False

    checkpoint = torch.load(
        Path(checkpoint_path) / "moe_adapter.pt", map_location=device, weights_only=False
    )
    moe_config = checkpoint["config"]

    moe_model = MoELoRAModel(base_model, moe_config)

    for layer_name, state_dict in checkpoint["moe_layers"].items():
        if layer_name in moe_model.moe_layers:
            adapter_state = {
                k: v for k, v in state_dict.items() if not k.startswith("base_layer.")
            }
            moe_model.moe_layers[layer_name].load_state_dict(adapter_state, strict=False)
            logger.info(f"Loaded MoE layer: {layer_name}")

    return moe_model.to(device)


def print_results_table(results_dict: dict[str, dict[str, float]]):
    """Print results in a formatted table."""
    datasets = [
        "ARC-Challenge",
        "ARC-Easy",
        "Winogrande",
        "BoolQ",
        "OpenBookQA",
        "HellaSwag",
    ]

    print("\n" + "=" * 90)
    print("| Dataset      | Base Acc. | Single LoRA Acc. | MoE Acc. |")
    print("|--------------|-----------|------------------|----------|")

    for dataset in datasets:
        base_acc = results_dict.get("base", {}).get(dataset, 0.0)
        lora_acc = results_dict.get("lora", {}).get(dataset, 0.0)
        moe_acc = results_dict.get("moe", {}).get(dataset, 0.0)

        print(f"| {dataset:12} | {base_acc:9.4f} | {lora_acc:16.4f} | {moe_acc:9.4f} |")

    print("|" + "-" * 88 + "|")

    base_avg = results_dict.get("base", {}).get("Average", 0.0)
    lora_avg = results_dict.get("lora", {}).get("Average", 0.0)
    moe_avg = results_dict.get("moe", {}).get("Average", 0.0)

    print(f"| {'Average':12} | {base_avg:9.4f} | {lora_avg:16.4f} | {moe_avg:9.4f} |")
    print("=" * 90)


def main():
    parser = get_benchmark_arg_parser()
    args = parser.parse_args()

    tokenizer = get_tokenizer(model_name=args.model_name, trust_remote_code=True)

    all_results = {}

    if not args.skip_base:
        logger.info("\n" + "=" * 50)
        logger.info("EVALUATING BASE MODEL")
        logger.info("=" * 50)
        base_model = load_base_model(args.model_name, args.device)
        base_evaluator = BenchmarkEvaluator(
            base_model, tokenizer, args.device, args.batch_size
        )
        all_results["base"] = base_evaluator.run_all_benchmarks()
        del base_model
        torch.cuda.empty_cache()
    else:
        logger.info("Skipping base model evaluation")

    if not args.skip_lora:
        logger.info("\n" + "=" * 50)
        logger.info("EVALUATING SINGLE LORA MODEL")
        logger.info("=" * 50)
        lora_model = load_lora_model(args.model_name, args.lora_checkpoint, args.device)
        lora_evaluator = BenchmarkEvaluator(
            lora_model, tokenizer, args.device, args.batch_size
        )
        all_results["lora"] = lora_evaluator.run_all_benchmarks()
        del lora_model
        torch.cuda.empty_cache()
    else:
        logger.info("Skipping LoRA model evaluation")

    if not args.skip_moe:
        logger.info("\n" + "=" * 50)
        logger.info("EVALUATING MOE MODEL")
        logger.info("=" * 50)
        moe_model = load_moe_model(args.model_name, args.moe_checkpoint, args.device)

        # Use very small batch size for MoE to avoid OOM
        # Process 1 question at a time (1 question × 4 options = 4 sequences max)
        moe_batch_size = 1
        logger.info(f"Using batch_size={moe_batch_size} for MoE model to prevent OOM")
        logger.info(
            "This will process 1 question at a time with max 4 sequences per forward pass"
        )

        moe_evaluator = BenchmarkEvaluator(
            moe_model,
            tokenizer,
            args.device,
            batch_size=moe_batch_size,  # Process 1 question at a time
            max_options_per_forward=4,  # Max 4 sequences per forward pass
        )
        all_results["moe"] = moe_evaluator.run_all_benchmarks()
        del moe_model
        torch.cuda.empty_cache()
    else:
        logger.info("Skipping MoE model evaluation")

    print_results_table(all_results)


if __name__ == "__main__":
    main()
