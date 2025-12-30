import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from peft import PeftModel
from service.moe import MoELoRAModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from service.model import get_model, get_tokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BenchmarkEvaluator:
    """Evaluator for multiple choice benchmarks."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda",
        batch_size: int = 8,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.model.eval()

    def evaluate_arc_challenge(self) -> float:
        """Evaluate on ARC-Challenge dataset."""
        logger.info("Evaluating ARC-Challenge...")
        dataset = load_dataset("ai2_arc", "ARC-Challenge", split="test")
        return self._evaluate_arc_style(dataset)

    def evaluate_arc_easy(self) -> float:
        """Evaluate on ARC-Easy dataset."""
        logger.info("Evaluating ARC-Easy...")
        dataset = load_dataset("ai2_arc", "ARC-Easy", split="test")
        return self._evaluate_arc_style(dataset)

    def evaluate_winogrande(self) -> float:
        """Evaluate on Winogrande dataset."""
        logger.info("Evaluating Winogrande...")
        dataset = load_dataset("winogrande", "winogrande_xl", split="validation")

        correct = 0
        total = 0

        for i in tqdm(range(0, len(dataset), self.batch_size), desc="Winogrande"):
            batch = dataset[i : i + self.batch_size]

            for sentence, option1, option2, answer in zip(
                batch["sentence"], batch["option1"], batch["option2"], batch["answer"]
            ):
                # Replace underscore with each option
                text1 = sentence.replace("_", option1)
                text2 = sentence.replace("_", option2)

                # Calculate perplexity for each option
                ppl1 = self._calculate_perplexity(text1)
                ppl2 = self._calculate_perplexity(text2)

                # Lower perplexity is better
                predicted = "1" if ppl1 < ppl2 else "2"

                if predicted == answer:
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0.0
        logger.info(f"Winogrande Accuracy: {accuracy:.4f}")
        return accuracy

    def evaluate_boolq(self) -> float:
        """Evaluate on BoolQ dataset."""
        logger.info("Evaluating BoolQ...")
        dataset = load_dataset("boolq", split="validation")

        correct = 0
        total = 0

        for i in tqdm(range(0, len(dataset), self.batch_size), desc="BoolQ"):
            batch = dataset[i : i + self.batch_size]

            for passage, question, answer in zip(
                batch["passage"], batch["question"], batch["answer"]
            ):
                prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer:"

                # Get probabilities for "Yes" and "No"
                yes_prob = self._get_token_probability(prompt, " Yes")
                no_prob = self._get_token_probability(prompt, " No")

                predicted = yes_prob > no_prob

                if predicted == answer:
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0.0
        logger.info(f"BoolQ Accuracy: {accuracy:.4f}")
        return accuracy

    def evaluate_openbookqa(self) -> float:
        """Evaluate on OpenBookQA dataset."""
        logger.info("Evaluating OpenBookQA...")
        dataset = load_dataset("openbookqa", "main", split="test")

        correct = 0
        total = 0

        for i in tqdm(range(0, len(dataset), self.batch_size), desc="OpenBookQA"):
            batch = dataset[i : i + self.batch_size]

            for question_stem, choices, answer_key in zip(
                batch["question_stem"], batch["choices"], batch["answerKey"]
            ):
                question = question_stem
                options = choices["text"]
                labels = choices["label"]

                # Calculate perplexity for each option
                perplexities = []
                for option in options:
                    prompt = f"Question: {question}\nAnswer: {option}"
                    ppl = self._calculate_perplexity(prompt)
                    perplexities.append(ppl)

                # Get the option with lowest perplexity
                predicted_idx = perplexities.index(min(perplexities))
                predicted_label = labels[predicted_idx]

                if predicted_label == answer_key:
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0.0
        logger.info(f"OpenBookQA Accuracy: {accuracy:.4f}")
        return accuracy

    def evaluate_hellaswag(self) -> float:
        """Evaluate on HellaSwag dataset."""
        logger.info("Evaluating HellaSwag...")
        dataset = load_dataset("hellaswag", split="validation")

        correct = 0
        total = 0

        for i in tqdm(range(0, len(dataset), self.batch_size), desc="HellaSwag"):
            batch = dataset[i : i + self.batch_size]

            for ctx, endings, label in zip(batch["ctx"], batch["endings"], batch["label"]):
                # Calculate perplexity for each ending
                perplexities = []
                for ending in endings:
                    full_text = ctx + " " + ending
                    ppl = self._calculate_perplexity(full_text)
                    perplexities.append(ppl)

                # Get the ending with lowest perplexity
                predicted_idx = perplexities.index(min(perplexities))

                if predicted_idx == int(label):
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0.0
        logger.info(f"HellaSwag Accuracy: {accuracy:.4f}")
        return accuracy

    def _evaluate_arc_style(self, dataset) -> float:
        """Helper function for ARC-style datasets."""
        correct = 0
        total = 0

        for i in tqdm(range(0, len(dataset), self.batch_size), desc="ARC"):
            batch = dataset[i : i + self.batch_size]

            for question, choices, answer_key in zip(
                batch["question"], batch["choices"], batch["answerKey"]
            ):
                options = choices["text"]
                labels = choices["label"]

                # Calculate perplexity for each option
                perplexities = []
                for option in options:
                    prompt = f"Question: {question}\nAnswer: {option}"
                    ppl = self._calculate_perplexity(prompt)
                    perplexities.append(ppl)

                # Get the option with lowest perplexity
                predicted_idx = perplexities.index(min(perplexities))
                predicted_label = labels[predicted_idx]

                if predicted_label == answer_key:
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0.0
        return accuracy

    def _calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity for a given text."""
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)

            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()

        return perplexity

    def _get_token_probability(self, prompt: str, token: str) -> float:
        """Get probability of a specific token following a prompt."""
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)

            # Get logits for the last token position
            last_logits = outputs.logits[0, -1, :]
            probs = torch.softmax(last_logits, dim=-1)

            # Get probability of the target token
            token_id = self.tokenizer.encode(token, add_special_tokens=False)[0]
            token_prob = probs[token_id].item()

        return token_prob

    def run_all_benchmarks(self) -> dict[str, float]:
        """Run all benchmark evaluations."""
        results = {
            "ARC-Challenge": self.evaluate_arc_challenge(),
            "ARC-Easy": self.evaluate_arc_easy(),
            "Winogrande": self.evaluate_winogrande(),
            "BoolQ": self.evaluate_boolq(),
            "OpenBookQA": self.evaluate_openbookqa(),
            "HellaSwag": self.evaluate_hellaswag(),
        }

        # Calculate average
        results["Average"] = sum(results.values()) / len(results)

        return results


def load_base_model(model_name: str, device: str) -> PreTrainedModel:
    """Load base model without any adapters."""
    logger.info(f"Loading base model: {model_name}")
    model = get_model(
        model_name=model_name,
        load_in_4bit=False,
        torch_dtype=torch.bfloat16,
        use_flash_attention=False,
    )
    return model.to(device)


def load_lora_model(
    model_name: str,
    checkpoint_path: str,
    device: str,
) -> PeftModel:
    """Load LoRA model from checkpoint."""
    logger.info(f"Loading LoRA model from: {checkpoint_path}")
    base_model = get_model(
        model_name=model_name,
        load_in_4bit=False,
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

    # Load base model
    base_model = get_model(
        model_name=model_name,
        load_in_4bit=False,
        torch_dtype=torch.bfloat16,
        use_flash_attention=False,
    )

    # Load MoE state
    checkpoint = torch.load(Path(checkpoint_path) / "moe_adapter.pt")
    moe_config = checkpoint["config"]

    # Create MoE model
    moe_model = MoELoRAModel(base_model, moe_config)

    # Load MoE layer states
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
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.model_name, trust_remote_code=True)

    all_results = {}

    # Evaluate base model
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

    # Evaluate LoRA model
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

    # Evaluate MoE model
    if not args.skip_moe:
        logger.info("\n" + "=" * 50)
        logger.info("EVALUATING MOE MODEL")
        logger.info("=" * 50)

        moe_model = load_moe_model(args.model_name, args.moe_checkpoint, args.device)
        moe_evaluator = BenchmarkEvaluator(
            moe_model, tokenizer, args.device, args.batch_size
        )
        all_results["moe"] = moe_evaluator.run_all_benchmarks()

        del moe_model
        torch.cuda.empty_cache()

    # Print results table
    print_results_table(all_results)


if __name__ == "__main__":
    main()
