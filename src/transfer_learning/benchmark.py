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
    """Print results in a formatted table.

    Args:
        results_dict: Dictionary with model types as keys ("base", "lora", "moe")
                     and their results dictionaries as values.
                     Each results dictionary has dataset names as keys and accuracy as values.

    Example:
        results_dict = {
            "base": {"ARC-Challenge": 0.45, "BoolQ": 0.81, "Average": 0.63},
            "lora": {"ARC-Challenge": 0.48, "BoolQ": 0.75, "Average": 0.62},
            "moe": {"ARC-Challenge": 0.55, "BoolQ": 0.80, "Average": 0.68},
        }

    """
    all_datasets = set()
    for model_results in results_dict.values():
        all_datasets.update(key for key in model_results.keys() if key != "Average")

    available_datasets = sorted(all_datasets)

    if not available_datasets:
        print("No datasets found in results!")
        return

    max_dataset_len = max(len(dataset) for dataset in available_datasets)
    dataset_col_width = max(max_dataset_len, 12)

    separator_width = dataset_col_width + 3 + 11 + 18 + 10 + 6  # columns + separators

    print("\n" + "=" * separator_width)
    print(f"| {'Dataset':<{dataset_col_width}} | Base Acc. | Single LoRA Acc. | MoE Acc. |")
    print(f"|{'-' * dataset_col_width}--|-----------|------------------|----------|")

    for dataset in available_datasets:
        base_acc = results_dict.get("base", {}).get(dataset, 0.0)
        lora_acc = results_dict.get("lora", {}).get(dataset, 0.0)
        moe_acc = results_dict.get("moe", {}).get(dataset, 0.0)
        print(
            f"| {dataset:<{dataset_col_width}} | {base_acc:9.4f} | {lora_acc:16.4f} | {moe_acc:9.4f} |"
        )

    print(f"|{'-' * (separator_width - 2)}|")

    base_avg = results_dict.get("base", {}).get("Average", 0.0)
    lora_avg = results_dict.get("lora", {}).get("Average", 0.0)
    moe_avg = results_dict.get("moe", {}).get("Average", 0.0)
    print(
        f"| {'Average':<{dataset_col_width}} | {base_avg:9.4f} | {lora_avg:16.4f} | {moe_avg:9.4f} |"
    )
    print("=" * separator_width)


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

        moe_evaluator = BenchmarkEvaluator(
            moe_model,
            tokenizer,
            args.device,
            batch_size=args.batch_size,
        )
        all_results["moe"] = moe_evaluator.run_all_benchmarks()
        del moe_model
        torch.cuda.empty_cache()
    else:
        logger.info("Skipping MoE model evaluation")

    print_results_table(all_results)


if __name__ == "__main__":
    main()
