import logging
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

from config.args import get_train_args_parser
from config.train import TrainingConfig
from lora.train import train_epoch, validate
from service.dataset import create_dataloaders
from service.model import get_model, get_tokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = get_train_args_parser()
    args = parser.parse_args()

    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights=True,
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

    output_path = Path(training_config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Single LoRA Configuration (PEFT):")
    logger.info(f"  Model: {training_config.model_name}")
    logger.info(f"  Rank: {peft_config.r}")
    logger.info(f"  Alpha: {peft_config.lora_alpha}")
    logger.info(f"  Dropout: {peft_config.lora_dropout}")
    logger.info(f"  Target Modules: {peft_config.target_modules}")
    logger.info(f"  Output Dir: {training_config.output_dir}")
    logger.info("=" * 80)

    logger.info("Loading tokenizer...")
    tokenizer = get_tokenizer(
        model_name=training_config.model_name,
        trust_remote_code=True,
    )

    logger.info("Loading base model...")
    base_model = get_model(
        model_name=training_config.model_name,
        load_in_4bit=training_config.load_in_4bit,
        torch_dtype=torch.bfloat16,
        use_flash_attention=training_config.use_flash_attention,
    )

    logger.info("Creating PEFT model with LoRA adapters...")
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    train_dataloader, val_dataloader = create_dataloaders(
        tokenizer=tokenizer,
        config=training_config,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
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
            lora_model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            max_grad_norm=1.0,
            logging_steps=10,
            device="cuda",
            epoch=epoch,
            num_epochs=training_config.num_epochs,
        )

        logger.info(f"Training Loss: {train_loss:.4f}")

        val_loss = validate(
            lora_model=model,
            val_dataloader=val_dataloader,
            device="cuda",
        )
        logger.info(f"Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_path / "single_lora"
            checkpoint_path.mkdir(exist_ok=True)

            model.save_pretrained(str(checkpoint_path))
            tokenizer.save_pretrained(checkpoint_path)

            torch.save(
                {
                    "peft_config": peft_config,
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
