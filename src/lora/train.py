import logging

import torch
from peft import PeftMixedModel, PeftModel
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def train_epoch(
    lora_model: PeftModel | PeftMixedModel,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler | None = None,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    logging_steps: int = 10,
    device: str = "cuda",
    epoch: int = 0,
    num_epochs: int = 1,
) -> float:
    """Train for one epoch."""
    lora_model.train()
    total_loss = 0.0
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

        outputs = lora_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss: Tensor = outputs.loss

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()

        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1

        if (step + 1) % gradient_accumulation_steps == 0:
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(lora_model.parameters(), max_grad_norm)

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad()

            if step % logging_steps == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
                        "lr": f"{current_lr:.2e}",
                    }
                )

    avg_loss = total_loss / num_batches

    logger.info("\nEpoch %d Summary:", epoch + 1)
    logger.info("  Training Loss: %.4f", avg_loss)

    return avg_loss


def validate(
    lora_model: PeftModel | PeftMixedModel,
    val_dataloader: DataLoader,
    device: str = "cuda",
) -> float:
    """Validate the model and return average loss."""
    lora_model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        progress_bar = tqdm(val_dataloader, desc="Validation")

        for batch in progress_bar:
            input_ids: Tensor = batch["input_ids"].to(device)
            attention_mask: Tensor = batch["attention_mask"].to(device)
            labels: Tensor = batch["labels"].to(device)

            outputs = lora_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss: Tensor = outputs.loss
            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches
    return avg_loss
