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
    return get_model(
        model_name=model_name,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        use_flash_attention=False,
        device_map=device,
    )


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

    checkpoint = torch.load(
        Path(checkpoint_path) / "moe_adapter.pt", map_location=device, weights_only=False
    )
    moe_config = checkpoint["config"]

    moe_model = MoELoRAModel(base_model, moe_config)

    for layer_name, state_dict in checkpoint["moe_layers"].items():
        if layer_name in moe_model.moe_layers:
            moe_model.moe_layers[layer_name].load_state_dict(state_dict)

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

    logger.info("\n" + "=" * 50)
    logger.info("EVALUATING BASE MODEL")
    logger.info("=" * 50)

    base_model = load_base_model(args.model_name, args.device)
    base_evaluator = BenchmarkEvaluator(base_model, tokenizer, args.device, args.batch_size)
    all_results["base"] = base_evaluator.run_all_benchmarks()

    del base_model
    torch.cuda.empty_cache()

    logger.info("\n" + "=" * 50)
    logger.info("EVALUATING SINGLE LORA MODEL")
    logger.info("=" * 50)

    lora_model = load_lora_model(args.model_name, args.lora_checkpoint, args.device)
    lora_evaluator = BenchmarkEvaluator(lora_model, tokenizer, args.device, args.batch_size)
    all_results["lora"] = lora_evaluator.run_all_benchmarks()

    del lora_model
    torch.cuda.empty_cache()

    logger.info("\n" + "=" * 50)
    logger.info("EVALUATING MOE MODEL")
    logger.info("=" * 50)

    moe_model = load_moe_model(args.model_name, args.moe_checkpoint, args.device)
    moe_evaluator = BenchmarkEvaluator(moe_model, tokenizer, args.device, args.batch_size)
    all_results["moe"] = moe_evaluator.run_all_benchmarks()

    del moe_model
    torch.cuda.empty_cache()

    print_results_table(all_results)


if __name__ == "__main__":
    main()
