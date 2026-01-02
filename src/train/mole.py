import logging
from pathlib import Path

import torch

from config.args import get_train_args_parser
from config.mole import MoLELoRAConfig
from config.train import TrainingConfig
from mole.model import MoELoRAModel
from mole.train import train_epoch, validate
from service.dataset import create_dataloaders
from service.model import get_model, get_tokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = get_train_args_parser()
    args = parser.parse_args()

    moe_config = MoLELoRAConfig(
        r=args.rank,
        num_experts=args.num_experts,
        top_k=args.top_k,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        adapter_init=args.adapter_init,
        router_init=args.router_init,
        target_modules=args.target_modules,
    )

    training_config = TrainingConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        warmup_steps=args.warmup_steps,
        load_in_4bit=args.load_in_4bit,
        use_flash_attention=args.use_flash_attention,
    )

    normalized_model_name = args.model_name.replace("/", "_")
    output_path = Path(normalized_model_name, training_config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("MoE LoRA Configuration:")
    logger.info(f"  Model: {training_config.model_name}")
    logger.info(f"  Experts: {moe_config.num_experts}")
    logger.info(f"  Top-K: {moe_config.top_k}")
    logger.info(f"  Rank: {moe_config.r}")
    logger.info(f"  Adapter Init: {moe_config.adapter_init}")
    logger.info(f"  Router Init: {moe_config.router_init}")
    logger.info(f"  Target Modules: {moe_config.target_modules}")
    logger.info("=" * 80)

    logger.info("Loading tokenizer...")
    tokenizer = get_tokenizer(
        model_name=training_config.model_name,
        trust_remote_code=True,
    )

    logger.info("Loading base model...")
    model = get_model(
        model_name=training_config.model_name,
        load_in_4bit=training_config.load_in_4bit,
        torch_dtype=torch.bfloat16,
        use_flash_attention=training_config.use_flash_attention,
    )

    train_dataloader, val_dataloader = create_dataloaders(
        tokenizer=tokenizer,
        config=training_config,
    )

    logger.info("Injecting MoE LoRA layers...")
    moe_model = MoELoRAModel(model, moe_config)
    moe_model.print_trainable_parameters()

    for param in model.parameters():
        param.requires_grad = False

    for param in moe_model.get_trainable_parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        moe_model.get_trainable_parameters(),
        lr=training_config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    total_steps = (
        len(train_dataloader)
        * training_config.num_epochs
        // training_config.gradient_accumulation_steps
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=training_config.learning_rate * 0.1,
    )

    logger.info("Starting training...")
    best_val_loss = float("inf")

    for epoch in range(training_config.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{training_config.num_epochs}")

        train_loss = train_epoch(
            moe_model=moe_model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            max_grad_norm=1.0,
            logging_steps=10,
            diversity_loss_weight=args.diversity_loss_coefficient,
            device="cuda",
            epoch=epoch,
            num_epochs=training_config.num_epochs,
        )

        logger.info(f"Training Loss: {train_loss:.4f}")

        val_loss = validate(
            moe_model=moe_model,
            val_dataloader=val_dataloader,
            device="cuda",
        )
        logger.info(f"Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_path.joinpath(
                f"{args.adapter_init}_{args.router_init}_{args.top_k}"
            )
            checkpoint_path.mkdir(exist_ok=True)

            moe_model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)

            torch.save(
                {
                    "moe_config": moe_config,
                    "training_config": training_config,
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                },
                checkpoint_path / "training_state.pt",
            )

            logger.info(f"Saved best model to {checkpoint_path}")

    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
