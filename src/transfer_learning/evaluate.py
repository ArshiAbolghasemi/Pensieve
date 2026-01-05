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
            self.max_options_per_forward = 64
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

    def evaluate_piqa(self) -> float:
        """Evaluate on PIQA dataset."""
        logger.info("Evaluating PIQA...")
        dataset = cast("Dataset", load_dataset("nthngdy/piqa", split="validation"))
        return self._evaluate_dataset(dataset, "piqa")

    def evaluate_swag(self) -> float:
        """Evaluate on SWAG dataset."""
        logger.info("Evaluating SWAG...")
        dataset = cast("Dataset", load_dataset("allenai/swag", split="validation"))
        return self._evaluate_dataset(dataset, "swag")

    def evaluate_storycloze(self) -> float:
        """Evaluate on StoryCloze Test dataset."""
        logger.info("Evaluating StoryCloze...")
        dataset = cast("Dataset", load_dataset("lecslab/story_cloze", split="test"))
        return self._evaluate_dataset(dataset, "storycloze")

    def evaluate_axb(self) -> float:
        """Evaluate on Broadcoverage Diagnostics (SuperGLUE) dataset."""
        logger.info("Evaluating AXB...")
        dataset = cast("Dataset", load_dataset("aps/super_glue", "axb", split="test"))
        return self._evaluate_dataset(dataset, "axb")

    def evaluate_axg(self) -> float:
        """Evaluate on Winogender Schema Diagnostics (SuperGLUE) dataset."""
        logger.info("Evaluating AXG...")
        dataset = cast("Dataset", load_dataset("aps/super_glue", "axg", split="test"))
        return self._evaluate_dataset(dataset, "axg")

    def evaluate_cb(self) -> float:
        """Evaluate on CommitmentBank (SuperGLUE) dataset."""
        logger.info("Evaluating CB...")
        dataset = cast("Dataset", load_dataset("aps/super_glue", "cb", split="validation"))
        return self._evaluate_dataset(dataset, "cb")

    def evaluate_copa(self) -> float:
        """Evaluate on COPA (SuperGLUE) dataset."""
        logger.info("Evaluating COPA...")
        dataset = cast(
            "Dataset", load_dataset("aps/super_glue", "copa", split="validation")
        )
        return self._evaluate_dataset(dataset, "copa")

    def evaluate_multirc(self) -> float:
        """Evaluate on MultiRC (SuperGLUE) dataset."""
        logger.info("Evaluating MultiRC...")
        dataset = cast(
            "Dataset", load_dataset("aps/super_glue", "multirc", split="validation")
        )
        return self._evaluate_dataset(dataset, "multirc")

    def evaluate_record(self) -> float:
        """Evaluate on ReCoRD (SuperGLUE) dataset."""
        logger.info("Evaluating ReCoRD...")
        dataset = cast(
            "Dataset", load_dataset("aps/super_glue", "record", split="validation")
        )
        return self._evaluate_dataset(dataset, "record")

    def evaluate_rte(self) -> float:
        """Evaluate on RTE (SuperGLUE) dataset."""
        logger.info("Evaluating RTE...")
        dataset = cast("Dataset", load_dataset("aps/super_glue", "rte", split="validation"))
        return self._evaluate_dataset(dataset, "rte")

    def evaluate_wic(self) -> float:
        """Evaluate on WiC (SuperGLUE) dataset."""
        logger.info("Evaluating WiC...")
        dataset = cast("Dataset", load_dataset("aps/super_glue", "wic", split="validation"))
        return self._evaluate_dataset(dataset, "wic")

    def evaluate_wsc(self) -> float:
        """Evaluate on WSC (SuperGLUE) dataset."""
        logger.info("Evaluating WSC...")
        dataset = cast(
            "Dataset", load_dataset("aps/super_glue", "wsc.fixed", split="validation")
        )
        return self._evaluate_dataset(dataset, "wsc")

    # ======================================================

    def _evaluate_dataset(self, dataset: Dataset, dataset_name: str) -> float:
        """Evaluate dataset using batched masked token loss approach."""
        labels = []
        predictions = []
        answer_sets = [] if dataset_name == "record" else None

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
                    if dataset_name == "record":
                        # For ReCoRD, store the set of valid answer indices
                        valid_indices = [
                            i
                            for i, entity in enumerate(row["entities"])
                            if entity in row["answers"]
                        ]
                        answer_sets.append(set(valid_indices))
                    else:
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
        if dataset_name == "record":
            # For ReCoRD, check if predicted index is in the set of valid answers
            correct = sum(
                1
                for pred, answer_set in zip(predictions, answer_sets)
                if pred in answer_set
            )
            accuracy = correct / len(predictions) if len(predictions) > 0 else 0.0
        else:
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
        if dataset_name == "piqa":
            return f"Given the goal: {row['goal']}"
        if dataset_name == "swag":
            return row["startphrase"]
        if dataset_name == "storycloze":
            return f"Given the text: {row['prompt']}"
        if dataset_name == "axb":
            return f"Sentence 1: {row['sentence1']}\nSentence 2: {row['sentence2']}\nDoes Sentence 1 entail Sentence 2?"
        if dataset_name == "axg":
            return f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}\nDoes the premise entail the hypothesis?"
        if dataset_name == "cb":
            return f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}\nDoes the premise entail the hypothesis?"
        if dataset_name == "copa":
            question_type = "cause" if row["question"] == "cause" else "effect"
            return f"Premise: {row['premise']}\n\nQuestion: What is the {question_type}?\nChoice 1: {row['choice1']}\nChoice 2: {row['choice2']}"
        if dataset_name == "multirc":
            return f"According to the Passage: {row['paragraph']}\nIs this Answer: {row['answer']}\n correct for this Question: {row['question']}"
        if dataset_name == "record":
            return f'Given the passage: {row["passage"]}\n\nThe right choice for "@placeholder" in the query "{row["query"]}" is:'
        if dataset_name == "rte":
            return f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}\nDoes the premise entail the hypothesis?"
        if dataset_name == "wic":
            s1 = row["sentence1"]
            s2 = row["sentence2"]
            word = row["word"]
            start1, end1 = row["start1"], row["end1"]
            start2, end2 = row["start2"], row["end2"]

            s1_marked = s1[:start1] + "*" + s1[start1:end1] + "*" + s1[end1:]
            s2_marked = s2[:start2] + "*" + s2[start2:end2] + "*" + s2[end2:]

            return f"Sentence 1: {s1_marked}\nSentence 2: {s2_marked}\n\nQuestion: Is the word *{word}* used in the same meaning in both sentences?"
        if dataset_name == "wsc":
            text = row["text"]
            span1_text = row["span1_text"]
            span2_text = row["span2_text"]
            span1_idx = row["span1_index"]
            span2_idx = row["span2_index"]

            if span2_idx > span1_idx:
                marked_text = (
                    text[:span2_idx]
                    + "["
                    + span2_text
                    + "]"
                    + text[span2_idx + len(span2_text) :]
                )
                marked_text = (
                    marked_text[:span1_idx]
                    + "*"
                    + span1_text
                    + "*"
                    + marked_text[span1_idx + len(span1_text) :]
                )
            else:
                marked_text = (
                    text[:span1_idx]
                    + "*"
                    + span1_text
                    + "*"
                    + text[span1_idx + len(span1_text) :]
                )
                offset = len(span1_text) + 2
                new_span2_idx = span2_idx if span2_idx < span1_idx else span2_idx + offset
                marked_text = (
                    marked_text[:new_span2_idx]
                    + "["
                    + span2_text
                    + "]"
                    + marked_text[new_span2_idx + len(span2_text) :]
                )

            return f"Text: {marked_text}\n\nQuestion: Does the pronoun/reference [{span2_text}] refer to *{span1_text}*?"
        # ======================================================
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
        if dataset_name in ["arc-challenge", "arc-easy"]:
            return [f"The right answer is: {choice}" for choice in row["choices"]["text"]]
        if dataset_name == "openbookqa":
            return [f"The right answer is: {choice}" for choice in row["choices"]["text"]]
        if dataset_name == "hellaswag":
            return [f"{ending} is the correct continuation." for ending in row["endings"]]
        if dataset_name == "piqa":
            return [
                f"The proper solution to reach it is: {row['sol1']}",
                f"The proper solution to reach it is: {row['sol2']}",
            ]
        if dataset_name == "swag":
            return [row["ending0"], row["ending1"], row["ending2"], row["ending3"]]
        if dataset_name == "storycloze":
            return [
                f"{row['chosen']} is the coherent and suitable ending.",
                f"{row['rejected']} is the coherent and suitable ending.",
            ]
        if dataset_name == "axb":
            return [
                "Yes, the first sentence entails the second.",
                "No, the first sentence does not entail the second.",
            ]
        if dataset_name == "axg":
            return [
                "Yes, the premise entails the hypothesis.",
                "No, the premise does not entail the hypothesis.",
            ]
        if dataset_name == "cb":
            return [
                "Yes, the hypothesis is definitely true given the premise.",
                "No, the hypothesis contradicts the premise.",
                "It's unclear whether the hypothesis is true or false.",
            ]
        if dataset_name == "copa":
            question_type = "cause" if row["question"] == "cause" else "effect"
            return [
                f"Choice 1 is the correct {question_type}.",
                f"Choice 2 is the correct {question_type}.",
            ]
        if dataset_name == "multirc":
            return [
                "No, this answer is incorrect for the given question and passage.",
                "Yes, this answer is correct for the given question and passage.",
            ]
        if dataset_name == "record":
            return [entity for entity in row["entities"]]
        if dataset_name == "rte":
            return [
                "Yes, the premise entails the hypothesis.",
                "No, the premise does not entail the hypothesis.",
            ]
        if dataset_name == "wic":
            word = row["word"]
            return [
                f"No. The word *{word}* has a different meaning in sentence 1 and sentence 2.",
                f"Yes. The word *{word}* has the same meaning in sentence 1 and sentence 2.",
            ]
        if dataset_name == "wsc":
            return ["No.", "Yes."]
        # ======================================================
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
            answer = row["answer"]
            if isinstance(answer, str):
                return int(answer) - 1
            return answer - 1
        if dataset_name == "boolq":
            if "answer" in row:
                return 0 if row["answer"] else 1
            if "label" in row:
                return 0 if row["label"] else 1
            raise KeyError(
                f"BoolQ row has neither 'answer' nor 'label' field. Available fields: {list(row.keys())}"
            )
        if dataset_name == "openbookqa":
            return row["choices"]["label"].index(row["answerKey"])
        if dataset_name == "hellaswag":
            label = row["label"]
            if isinstance(label, str):
                return int(label)
            return label
        if dataset_name == "piqa":
            return int(row["label"])
        if dataset_name == "swag":
            return int(row["label"])
        if dataset_name == "storycloze":
            return 0  # 'chosen' is always the correct answer
        if dataset_name == "axb":
            return row["label"]
        if dataset_name == "axg":
            return row["label"]
        if dataset_name == "cb":
            return int(row["label"])
        if dataset_name == "copa":
            return int(row["label"])
        if dataset_name == "multirc":
            return row["label"]
        if dataset_name == "record":
            # Find index of correct entity from answers list
            for answer in row["answers"]:
                if answer in row["entities"]:
                    return row["entities"].index(answer)
            return 0  # Fallback
        if dataset_name == "rte":
            return int(row["label"])
        if dataset_name == "wic":
            return row["label"]
        if dataset_name == "wsc":
            return row["label"]
        # ======================================================
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
            "PIQA": self.evaluate_piqa(),
            "SWAG": self.evaluate_swag(),
            "StoryCloze": self.evaluate_storycloze(),
            "AXB": self.evaluate_axb(),
            "AXG": self.evaluate_axg(),
            "CB": self.evaluate_cb(),
            "COPA": self.evaluate_copa(),
            "MultiRC": self.evaluate_multirc(),
            "ReCoRD": self.evaluate_record(),
            "RTE": self.evaluate_rte(),
            "WiC": self.evaluate_wic(),
            "WSC": self.evaluate_wsc(),
        }
        results["Average"] = sum(results.values()) / len(results)
        return results

    def run_selected_benchmarks(self, benchmark_names: list[str]) -> dict[str, float]:
        """Run selected benchmark evaluations.

        Args:
            benchmark_names: List of benchmark names to run (e.g., ["arc-challenge", "boolq"])

        Returns:
            Dictionary mapping benchmark names to accuracy scores

        """
        benchmark_map = {
            "arc-challenge": self.evaluate_arc_challenge,
            "arc-easy": self.evaluate_arc_easy,
            "winogrande": self.evaluate_winogrande,
            "boolq": self.evaluate_boolq,
            "openbookqa": self.evaluate_openbookqa,
            "hellaswag": self.evaluate_hellaswag,
            "piqa": self.evaluate_piqa,
            "swag": self.evaluate_swag,
            "storycloze": self.evaluate_storycloze,
            "axb": self.evaluate_axb,
            "axg": self.evaluate_axg,
            "cb": self.evaluate_cb,
            "copa": self.evaluate_copa,
            "multirc": self.evaluate_multirc,
            "record": self.evaluate_record,
            "rte": self.evaluate_rte,
            "wic": self.evaluate_wic,
            "wsc": self.evaluate_wsc,
        }

        results = {}
        for name in benchmark_names:
            if name in benchmark_map:
                results[name] = benchmark_map[name]()
            else:
                logger.warning(f"Unknown benchmark: {name}")

        if results:
            results["Average"] = sum(results.values()) / len(results)

        return results
