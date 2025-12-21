import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel

from config.moe import MoELoRAConfig
from service.moe import MoELoRAModel


def train_moe_lora(
    model: PreTrainedModel,
    config: MoELoRAConfig,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    num_epochs: int = 3,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    logging_steps: int = 10,
    device: str = "cuda",
) -> PreTrainedModel:
    """Train MoE LoRA model with proper gradient accumulation and progress tracking.

    Args:
        model: Pretrained model to inject MoE LoRA layers into
        config: MoE LoRA configuration
        train_dataloader: Training data loader
        optimizer: Optimizer for training
        scheduler: Optional learning rate scheduler
        num_epochs: Number of training epochs
        gradient_accumulation_steps: Steps to accumulate gradients before update
        max_grad_norm: Maximum gradient norm for clipping
        logging_steps: Log every N steps
        device: Device to train on

    Returns:
        Trained model

    """
    moe_model = MoELoRAModel(model, config)
    moe_model.print_trainable_parameters()

    for param in model.parameters():
        param.requires_grad = False

    for param in moe_model.get_trainable_parameters():
        param.requires_grad = True

    model.train()
    global_step = 0

    for epoch in range(num_epochs):
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

            outputs = model(
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
                    torch.nn.utils.clip_grad_norm_(
                        moe_model.get_trainable_parameters(), max_grad_norm
                    )

                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                optimizer.zero_grad()

                global_step += 1

                if global_step % logging_steps == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    progress_bar.set_postfix(
                        {
                            "loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
                            "lr": f"{current_lr:.2e}",
                            "step": global_step,
                        }
                    )

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

    return model


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

    avg_loss = total_loss / num_batches
    return avg_loss
