import logging
from typing import cast

import torch
from datasets import Dataset, load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from mole.model import MoELoRAModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BenchmarkEvaluator:
    """Evaluator for multiple choice benchmarks."""

    def __init__(
        self,
        model: PreTrainedModel | PeftModel | MoELoRAModel,
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
        dataset = cast("Dataset", load_dataset("ai2_arc", "ARC-Challenge", split="test"))
        return self._evaluate_arc_style(dataset)

    def evaluate_arc_easy(self) -> float:
        """Evaluate on ARC-Easy dataset."""
        logger.info("Evaluating ARC-Easy...")
        dataset = cast("Dataset", load_dataset("ai2_arc", "ARC-Easy", split="test"))
        return self._evaluate_arc_style(dataset)

    def evaluate_winogrande(self) -> float:
        """Evaluate on Winogrande dataset."""
        logger.info("Evaluating Winogrande...")
        dataset = cast(
            "Dataset", load_dataset("winogrande", "winogrande_xl", split="validation")
        )

        correct = 0
        total = 0

        for i in tqdm(range(0, dataset.num_rows, self.batch_size), desc="Winogrande"):
            batch = dataset[i : i + self.batch_size]

            for sentence, option1, option2, answer in zip(
                batch["sentence"], batch["option1"], batch["option2"], batch["answer"]
            ):
                text1 = sentence.replace("_", option1)
                text2 = sentence.replace("_", option2)

                ppl1 = self._calculate_perplexity(text1)
                ppl2 = self._calculate_perplexity(text2)

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
        dataset = cast("Dataset", load_dataset("boolq", split="validation"))

        correct = 0
        total = 0

        for i in tqdm(range(0, dataset.num_rows, self.batch_size), desc="BoolQ"):
            batch = dataset[i : i + self.batch_size]

            for passage, question, answer in zip(
                batch["passage"], batch["question"], batch["answer"]
            ):
                prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer:"

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
        dataset = cast("Dataset", load_dataset("openbookqa", "main", split="test"))

        correct = 0
        total = 0

        for i in tqdm(range(0, dataset.num_rows, self.batch_size), desc="OpenBookQA"):
            batch = dataset[i : i + self.batch_size]

            for question_stem, choices, answer_key in zip(
                batch["question_stem"], batch["choices"], batch["answerKey"]
            ):
                question = question_stem
                options = choices["text"]
                labels = choices["label"]

                perplexities = []
                for option in options:
                    prompt = f"Question: {question}\nAnswer: {option}"
                    ppl = self._calculate_perplexity(prompt)
                    perplexities.append(ppl)

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
        dataset = cast("Dataset", load_dataset("hellaswag", split="validation"))

        correct = 0
        total = 0

        for i in tqdm(range(0, dataset.num_rows, self.batch_size), desc="HellaSwag"):
            batch = dataset[i : i + self.batch_size]

            for ctx, endings, label in zip(batch["ctx"], batch["endings"], batch["label"]):
                perplexities = []
                for ending in endings:
                    full_text = ctx + " " + ending
                    ppl = self._calculate_perplexity(full_text)
                    perplexities.append(ppl)

                predicted_idx = perplexities.index(min(perplexities))

                if predicted_idx == int(label):
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0.0
        logger.info(f"HellaSwag Accuracy: {accuracy:.4f}")
        return accuracy

    def _evaluate_arc_style(self, dataset: Dataset) -> float:
        """Helper function for ARC-style datasets."""
        correct = 0
        total = 0

        for i in tqdm(range(0, dataset.num_rows, self.batch_size), desc="ARC"):
            batch = dataset[i : i + self.batch_size]

            for question, choices, answer_key in zip(
                batch["question"], batch["choices"], batch["answerKey"]
            ):
                options = choices["text"]
                labels = choices["label"]

                perplexities = []
                for option in options:
                    prompt = f"Question: {question}\nAnswer: {option}"
                    ppl = self._calculate_perplexity(prompt)
                    perplexities.append(ppl)

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

            last_logits = outputs.logits[0, -1, :]
            probs = torch.softmax(last_logits, dim=-1)

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
        results["Average"] = sum(results.values()) / len(results)
        return results
