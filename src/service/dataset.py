import logging
from typing import cast

from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    BatchEncoding,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
)

from config.train import TrainingConfig

logger = logging.getLogger(__name__)


def _format_with_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    example: dict[str, str],
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


def _tokenize_for_causal_lm(
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    examples: dict[str, list[str]],
) -> BatchEncoding:
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    tokenized["labels"] = tokenized["input_ids"]
    return tokenized


def sample_flan_dataset(
    tokenizer: PreTrainedTokenizerBase,
    samples_per_task: int = 200,
) -> dict[str, Dataset]:
    """Load and format FLAN dataset for conversational SFT."""
    logger.info("Loading FLAN dataset...")
    dataset = cast("DatasetDict", load_dataset("TahaBa/flan-routing-MoE-dataset"))
    logger.info("Dataset loaded successfully")

    for split in ("train", "validation"):
        if split not in dataset:
            raise ValueError(f"{split} split is missing")

    expected_cols = {"source", "target"}
    actual_cols = set(dataset["train"].column_names)
    if not expected_cols.issubset(actual_cols):
        raise ValueError(f"Expected columns {expected_cols}, but got {actual_cols}")

    dataset = dataset.rename_columns(
        {"source": "prompt", "target": "completion"}
    ).select_columns(["prompt", "completion"])

    logger.info("Formatting dataset with chat template...")

    dataset = dataset.map(
        lambda ex: _format_with_chat_template(tokenizer, ex),
        remove_columns=dataset["train"].column_names,
    )

    sampled_train = (
        dataset["train"]
        .shuffle(seed=42)
        .select(range(min(samples_per_task, len(dataset["train"]))))
    )
    sampled_val = (
        dataset["validation"]
        .shuffle(seed=42)
        .select(range(min(samples_per_task, len(dataset["validation"]))))
    )

    logger.info("Train samples: %d", len(sampled_train))
    logger.info("Validation samples: %d", len(sampled_val))
    logger.info("Sample text: %s...", sampled_train[0]["text"][:200])

    return {"train": sampled_train, "validation": sampled_val}


def create_dataloaders(
    tokenizer: PreTrainedTokenizer,
    config: TrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders."""
    logger.info("Preparing datasets...")

    datasets = sample_flan_dataset(
        tokenizer=tokenizer,
        samples_per_task=config.samples_per_task,
    )

    logger.info("Tokenizing datasets...")

    train_dataset = datasets["train"].map(
        lambda ex: _tokenize_for_causal_lm(tokenizer, config.max_length, ex),
        batched=True,
        remove_columns=datasets["train"].column_names,
        desc="Tokenizing train dataset",
    )

    val_dataset = datasets["validation"].map(
        lambda ex: _tokenize_for_causal_lm(tokenizer, config.max_length, ex),
        batched=True,
        remove_columns=datasets["validation"].column_names,
        desc="Tokenizing validation dataset",
    )

    # Set format to PyTorch tensors for compatibility with DataLoader
    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    train_dataloader = DataLoader(
        cast("TorchDataset", train_dataset),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=4,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        cast("TorchDataset", val_dataset),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=4,
        pin_memory=True,
    )

    logger.info("Train batches: %d", len(train_dataloader))
    logger.info("Validation batches: %d", len(val_dataloader))

    return train_dataloader, val_dataloader
