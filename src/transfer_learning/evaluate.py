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
    """Evaluator for multiple choice benchmarks using masked token loss."""

    def __init__(
        self,
        model: PreTrainedModel | PeftModel | MoELoRAModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda",
        batch_size: int = 8,
        max_length: int = 512,
        max_options_per_forward: int | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.model.eval()

        if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

        self.is_moe_model = isinstance(self.model, MoELoRAModel)

        if max_options_per_forward is not None:
            self.max_options_per_forward = max_options_per_forward
        elif self.is_moe_model:
            # MoE models need very small chunks due to expert computation memory
            # With 4 sequences, expert_outputs tensor is ~500MB instead of 2GB+
            self.max_options_per_forward = 4
            logger.info(
                f"Detected MoE model - using max_options_per_forward={self.max_options_per_forward} to prevent OOM"
            )
        else:
            # Regular models can handle larger batches
            self.max_options_per_forward = 64

    def evaluate_arc_challenge(self) -> float:
        """Evaluate on ARC-Challenge dataset."""
        logger.info("Evaluating ARC-Challenge...")
        dataset = cast("Dataset", load_dataset("ai2_arc", "ARC-Challenge", split="test"))
        return self._evaluate_dataset(dataset, "arc-challenge")

    def evaluate_arc_easy(self) -> float:
        """Evaluate on ARC-Easy dataset."""
        logger.info("Evaluating ARC-Easy...")
        dataset = cast("Dataset", load_dataset("ai2_arc", "ARC-Easy", split="test"))
        return self._evaluate_dataset(dataset, "arc-easy")

    def evaluate_winogrande(self) -> float:
        """Evaluate on Winogrande dataset."""
        logger.info("Evaluating Winogrande...")
        dataset = cast(
            "Dataset", load_dataset("winogrande", "winogrande_xl", split="validation")
        )
        return self._evaluate_dataset(dataset, "winogrande")

    def evaluate_boolq(self) -> float:
        """Evaluate on BoolQ dataset."""
        logger.info("Evaluating BoolQ...")
        dataset = cast("Dataset", load_dataset("boolq", split="validation"))
        return self._evaluate_dataset(dataset, "boolq")

    def evaluate_openbookqa(self) -> float:
        """Evaluate on OpenBookQA dataset."""
        logger.info("Evaluating OpenBookQA...")
        dataset = cast("Dataset", load_dataset("openbookqa", "main", split="test"))
        return self._evaluate_dataset(dataset, "openbookqa")

    def evaluate_hellaswag(self) -> float:
        """Evaluate on HellaSwag dataset."""
        logger.info("Evaluating HellaSwag...")
        dataset = cast("Dataset", load_dataset("hellaswag", split="validation"))
        return self._evaluate_dataset(dataset, "hellaswag")

    def _evaluate_dataset(self, dataset: Dataset, dataset_name: str) -> float:
        """Evaluate dataset using batched masked token loss approach."""
        labels = []
        predictions = []

        for start in tqdm(
            range(0, len(dataset), self.batch_size),
            total=(len(dataset) + self.batch_size - 1) // self.batch_size,
            desc=f"Evaluating {dataset_name}",
        ):
            try:
                rows = [
                    dataset[i]
                    for i in range(start, min(start + self.batch_size, len(dataset)))
                ]

                # Build the flattened option texts for this batch
                all_texts = []
                options_per_sample = []
                ctx_lens_per_option = []

                for row in rows:
                    # Get formatted options using chat template
                    options = self._create_multi_choice_options(row, dataset_name)
                    options_per_sample.append(len(options))

                    # Compute context length
                    content = self._extract_input_content(row, dataset_name)
                    messages = [{"role": "user", "content": content}]
                    context_prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    ctx_len = len(self.tokenizer.encode(context_prompt)) - 1

                    all_texts.extend(options)
                    ctx_lens_per_option.extend([ctx_len] * len(options))

                    # Collect gold label
                    labels.append(self._extract_target_index(row, dataset_name))

                # Process in chunks to avoid OOM with MoE models
                all_losses = []
                for chunk_start in range(0, len(all_texts), self.max_options_per_forward):
                    chunk_end = min(
                        chunk_start + self.max_options_per_forward, len(all_texts)
                    )
                    chunk_texts = all_texts[chunk_start:chunk_end]
                    chunk_ctx_lens = ctx_lens_per_option[chunk_start:chunk_end]

                    # Tokenize chunk
                    tokenized = self.tokenizer(
                        chunk_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                    )
                    tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

                    # Create masked labels: ignore context and padding
                    masked_labels = tokenized["input_ids"].clone()
                    for i, ctx_len in enumerate(chunk_ctx_lens):
                        masked_labels[i, :ctx_len] = -100
                    masked_labels[tokenized["attention_mask"] == 0] = -100

                    with torch.no_grad():
                        try:
                            logits = self.model(
                                input_ids=tokenized["input_ids"],
                                attention_mask=tokenized["attention_mask"],
                                use_cache=False,
                            ).logits

                            # Compute per-sequence losses
                            chunk_losses = (
                                self._compute_loglike_loss(
                                    logits, masked_labels, reduction="none"
                                )
                                .detach()
                                .cpu()
                            )

                            all_losses.append(chunk_losses)

                        except torch.cuda.OutOfMemoryError:
                            # If even the chunk is too large, process one sequence at a time
                            logger.warning(
                                f"OOM with chunk size {len(chunk_texts)}, falling back to single-sequence processing"
                            )
                            single_losses = []
                            for seq_idx in range(len(chunk_texts)):
                                single_text = [chunk_texts[seq_idx]]
                                single_ctx_len = [chunk_ctx_lens[seq_idx]]

                                single_tokenized = self.tokenizer(
                                    single_text,
                                    return_tensors="pt",
                                    truncation=True,
                                    max_length=self.max_length,
                                ).to(self.device)

                                single_labels = single_tokenized["input_ids"].clone()
                                single_labels[0, : single_ctx_len[0]] = -100

                                single_logits = self.model(
                                    **single_tokenized, use_cache=False
                                ).logits
                                single_loss = (
                                    self._compute_loglike_loss(
                                        single_logits, single_labels, reduction="none"
                                    )
                                    .detach()
                                    .cpu()
                                )
                                single_losses.append(single_loss)

                                del single_tokenized, single_labels, single_logits
                                torch.cuda.empty_cache()

                            chunk_losses = torch.cat(single_losses, dim=0)
                            all_losses.append(chunk_losses)

                    # Clear GPU cache after each chunk for MoE models
                    if self.is_moe_model:
                        del tokenized, masked_labels
                        if "logits" in locals():
                            del logits
                        torch.cuda.empty_cache()

                # Concatenate all losses
                losses = torch.cat(all_losses, dim=0)

                # Reduce per sample (argmin across its options)
                idx = 0
                for n_opt in options_per_sample:
                    pred = torch.argmin(losses[idx : idx + n_opt]).item()
                    predictions.append(pred)
                    idx += n_opt

            except Exception as e:
                logger.error(
                    f"Error processing batch starting at index {start} for dataset {dataset_name}"
                )
                logger.error(f"Error details: {e}")
                if start < len(dataset):
                    logger.error(f"Problematic row example: {dataset[start]}")
                raise

        # Calculate accuracy
        correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
        accuracy = correct / len(labels) if len(labels) > 0 else 0.0
        logger.info(f"{dataset_name} Accuracy: {accuracy:.4f}")
        return accuracy

    def _extract_input_content(self, row, dataset_name: str) -> str:
        """Extract the input content/question from a dataset row."""
        if dataset_name in ["arc-challenge", "arc-easy"]:
            return f"Given the question: {row['question']}"
        if dataset_name == "winogrande":
            return f"Given the text: {row['sentence']}"
        if dataset_name == "boolq":
            return f"Passage: {row['passage']}\nQuestion: {row['question']}\nAnswer (True or False)?"
        if dataset_name == "openbookqa":
            return row["question_stem"]
        if dataset_name == "hellaswag":
            return row["ctx"]
        raise ValueError(f"Unknown dataset: {dataset_name}")

    def _get_verbalizer_choices(self, row, dataset_name: str) -> list[str] | None:
        """Get natural language verbalizers for classification tasks."""
        if dataset_name == "boolq":
            return ["Yes, that is true.", "No, that is false."]
        if dataset_name == "winogrande":
            return [
                f"{row['option1']} is the right choice for the placeholder _",
                f"{row['option2']} is the right choice for the placeholder _",
            ]
        if dataset_name in ["arc-challenge", "arc-easy"] or dataset_name == "openbookqa":
            return [f"The right answer is: {choice}" for choice in row["choices"]["text"]]
        if dataset_name == "hellaswag":
            return [f"{ending} is the correct continuation." for ending in row["endings"]]

        return None

    def _create_multi_choice_options(self, row, dataset_name: str) -> list[str]:
        """Create multi-choice options using the model's chat template."""
        options_texts = []
        content = self._extract_input_content(row, dataset_name)

        # Get natural verbalizer choices
        choices = self._get_verbalizer_choices(row, dataset_name)

        if choices is None:
            raise ValueError(f"No verbalizer defined for dataset: {dataset_name}")

        for choice in choices:
            # Use tokenizer's chat template to format the conversation
            messages = [
                {"role": "user", "content": content},
                {"role": "assistant", "content": choice},
            ]
            formatted_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            options_texts.append(formatted_text)

        return options_texts

    def _extract_target_index(self, row, dataset_name: str) -> int:
        """Extract the correct answer index from a dataset row."""
        if dataset_name in ["arc-challenge", "arc-easy"]:
            return row["choices"]["label"].index(row["answerKey"])
        if dataset_name == "winogrande":
            # Winogrande uses "answer" field with values "1" or "2"
            answer = row["answer"]
            # Handle both string and int types
            if isinstance(answer, str):
                return int(answer) - 1
            return answer - 1
        if dataset_name == "boolq":
            # BoolQ standard dataset uses 'answer' (boolean) not 'label'
            # True -> index 0 ("Yes, that is true.")
            # False -> index 1 ("No, that is false.")
            if "answer" in row:
                return 0 if row["answer"] else 1
            if "label" in row:
                # Fallback for SuperGLUE version
                return 0 if row["label"] else 1
            raise KeyError(
                f"BoolQ row has neither 'answer' nor 'label' field. Available fields: {list(row.keys())}"
            )
        if dataset_name == "openbookqa":
            return row["choices"]["label"].index(row["answerKey"])
        if dataset_name == "hellaswag":
            label = row["label"]
            # Handle both string and int types
            if isinstance(label, str):
                return int(label)
            return label
        raise ValueError(f"Unknown dataset: {dataset_name}")

    def _compute_loglike_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, reduction: str = "none"
    ) -> torch.Tensor:
        """Compute log-likelihood loss with optional masking."""
        bs = logits.size(0)
        vocab_size = logits.size(-1)
        labels = labels.squeeze(-1)

        # Shift for autoregressive prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction=reduction)
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)

        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

        # Reshape back and compute average per sequence
        if reduction == "none":
            loss = loss.view((bs, -1))
            non_zero_loss = (loss != 0).sum(dim=-1)
            non_zero_loss[non_zero_loss == 0] = 1
            loss = loss.sum(dim=-1) / non_zero_loss

        return loss.float()

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
