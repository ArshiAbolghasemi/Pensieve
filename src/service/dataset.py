import logging
from copy import copy
from typing import cast

from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    BatchEncoding,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
)

from config.train import TrainingConfig

logger = logging.getLogger(__name__)


def _format_with_chat_template(
    tokenizer: PreTrainedTokenizerBase, example: dict[str, str]
) -> dict[str, str]:
    messages = [
        {"role": "user", "content": example["prompt"].strip()},
        {"role": "assistant", "content": example["completion"].strip()},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": str(text)}


def sample_flan_dataset(
    tokenizer: PreTrainedTokenizerBase, samples_per_task: int = 200
) -> dict[str, Dataset]:
    """Load the FLAN dataset and sample a fixed number of examples per task.

    Format the dataset for conversational SFT training using a chat template.

    Args:
        samples_per_task (int): Number of samples per task.
        tokenizer: Tokenizer object with `apply_chat_template` method.

    Returns:
        dict: {"train": Dataset, "validation": Dataset or None}

    """
    logger.info("Loading FLAN dataset...")
    dataset = cast("DatasetDict", load_dataset("TahaBa/flan-routing-MoE-dataset"))
    logger.info("Dataset loaded successfully")

    if "train" not in dataset:
        msg = "train samples are missed"
        raise ValueError(msg)
    logger.info("Train samples len: %d", dataset.num_rows["train"])

    if "validation" not in dataset:
        msg = "validation samples are missed"
        raise ValueError(msg)

    expected_cols = {"source", "target"}
    actual_cols = set(dataset["train"].column_names)
    if not expected_cols.issubset(actual_cols):
        msg = (
            "Dataset columns do not match expected format. "
            f"Expected at least {expected_cols}, but got {actual_cols}."
        )
        raise ValueError(msg)

    logger.info(
        "Dataset has %s columns. Formatting for conversational SFT...", expected_cols
    )

    dataset_fmt = dataset.rename_columns(
        {"source": "prompt", "target": "completion"}
    ).select_columns(["prompt", "completion"])

    logger.info("Formatting dataset with chat template...")

    dataset_fmt = dataset_fmt.map(
        lambda example: _format_with_chat_template(tokenizer, example),
        remove_columns=dataset_fmt["train"].column_names,
    )

    sampled_train = (
        dataset_fmt["train"]
        .shuffle(seed=42)
        .select(range(min(samples_per_task, len(dataset_fmt["train"]))))
    )
    sampled_val = (
        dataset_fmt["validation"]
        .shuffle(seed=42)
        .select(range(min(samples_per_task, len(dataset_fmt["validation"]))))
    )

    logger.info("Training dataset prepared with %d samples", len(sampled_train))
    logger.info("Validation dataset prepared with %d samples", len(sampled_val))

    logger.info("Sample entry (first 200 chars):")
    logger.info("  - text: %s...", sampled_train[0]["text"][:200])

    return {"train": sampled_train, "validation": sampled_val}


def _tokenize_for_causal_lm(
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    examples: dict[str, list[str]],
) -> BatchEncoding:
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )
    tokenized["labels"] = copy(tokenized["input_ids"])
    return tokenized


def create_dataloaders(
    tokenizer: PreTrainedTokenizer,
    config: TrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders.

    Args:
        tokenizer: Tokenizer for processing text
        config: Training configuration with batch_size, max_length, etc.

    Returns:
        Tuple of (train_dataloader, val_dataloader)

    """
    logger.info("Loading and preparing dataset...")

    datasets = sample_flan_dataset(
        tokenizer=tokenizer, samples_per_task=config.samples_per_task
    )

    train_dataset = datasets["train"]
    val_dataset = datasets["validation"]

    logger.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda ex: _tokenize_for_causal_lm(tokenizer, config.max_length, ex),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset",
    )

    val_dataset = val_dataset.map(
        lambda ex: _tokenize_for_causal_lm(tokenizer, config.max_length, ex),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation dataset",
    )

    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_dataloader = DataLoader(
        cast("TorchDataset", train_dataset),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        cast("TorchDataset", val_dataset),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    logger.info(f"Train batches: {len(train_dataloader)}")
    logger.info(f"Validation batches: {len(val_dataloader)}")

    return train_dataloader, val_dataloader
