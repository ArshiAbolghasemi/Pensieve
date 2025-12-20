import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from config.moe import MoELoRAConfig
from service.moe import MoELoRAModel


def train_moe_lora(
    model: PreTrainedModel,
    config: MoELoRAConfig,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 3,
    device: str = "cuda",
) -> PreTrainedModel:
    moe_model = MoELoRAModel(model, config)
    moe_model.print_trainable_parameters()

    for param in model.parameters():
        param.requires_grad = False

    for param in moe_model.get_trainable_parameters():
        param.requires_grad = True

    model.train()

    for epoch in range(num_epochs):
        total_loss: float = 0.0

        for _, batch in enumerate(train_dataloader):
            input_ids: Tensor = batch["input_ids"].to(device)
            attention_mask: Tensor = batch["attention_mask"].to(device)
            labels: Tensor = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss: Tensor = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        avg_loss: float = total_loss / len(train_dataloader)
        print(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.4f}")

    return model
