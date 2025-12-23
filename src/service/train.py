import logging

import torch
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel

from service.moe import MoELoRAModel

logger = logging.getLogger(__name__)


def train_epoch(
    model: PreTrainedModel,
    moe_model: MoELoRAModel,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler | None = None,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    logging_steps: int = 10,
    diversity_loss_weight: float = 0.01,
    device: str = "cuda",
    epoch: int = 0,
    num_epochs: int = 1,
) -> float:
    """Train for one epoch with diversity regularization for MoE routing.

    Diversity loss encourages top-k selected experts to be different from each other
    by penalizing high cosine similarity between their weight representations.

    Args:
        model: Model to train
        moe_model: MoE model wrapper for accessing layers
        train_dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Optional learning rate scheduler
        gradient_accumulation_steps: Gradient accumulation steps
        max_grad_norm: Maximum gradient norm for clipping
        logging_steps: Log every N steps
        diversity_loss_weight: Weight for diversity regularization loss
        device: Device to train on
        epoch: Current epoch number (0-indexed)
        num_epochs: Total number of epochs

    Returns:
        Average training loss for the epoch

    """
    model.train()
    total_loss = 0.0
    total_task_loss = 0.0
    total_diversity_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(
        train_dataloader,
        desc=f"Epoch {epoch + 1}/{num_epochs}",
        total=len(train_dataloader),
    )

    for step, batch in enumerate(progress_bar):
        input_ids: Tensor = batch["input_ids"].to(device)
        attention_mask: Tensor = batch["attention_mask"].to(device)
        labels: Tensor = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        task_loss: Tensor = outputs.loss

        diversity_loss: Tensor = moe_model.compute_total_diversity_loss()

        loss = task_loss + diversity_loss_weight * diversity_loss

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()

        total_loss += loss.item() * gradient_accumulation_steps
        total_task_loss += task_loss.item()
        total_diversity_loss += diversity_loss.item()
        num_batches += 1

        if (step + 1) % gradient_accumulation_steps == 0:
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad()

            if step % logging_steps == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                progress_bar.set_postfix(
                    {
                        "total": f"{loss.item() * gradient_accumulation_steps:.4f}",
                        "task": f"{task_loss.item():.4f}",
                        "div": f"{diversity_loss.item():.4f}",
                        "lr": f"{current_lr:.2e}",
                    }
                )

    avg_loss = total_loss / num_batches
    avg_task_loss = total_task_loss / num_batches
    avg_diversity_loss = total_diversity_loss / num_batches

    logger.info("\nEpoch %d Summary:", epoch + 1)
    logger.info("  Total Loss: %.4f", avg_loss)
    logger.info("  Task Loss: %.4f", avg_task_loss)
    logger.info("  Diversity Loss: %.4f", avg_diversity_loss)

    return avg_loss


def validate(
    model: PreTrainedModel,
    val_dataloader: DataLoader,
    device: str = "cuda",
) -> float:
    """Validate the model and return average loss.

    Args:
        model: Model to validate
        val_dataloader: Validation data loader
        device: Device to run validation on

    Returns:
        Average validation loss

    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        progress_bar = tqdm(val_dataloader, desc="Validation")

        for batch in progress_bar:
            input_ids: Tensor = batch["input_ids"].to(device)
            attention_mask: Tensor = batch["attention_mask"].to(device)
            labels: Tensor = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss: Tensor = outputs.loss
            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / num_batches
