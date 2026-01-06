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
    """Main training function for MoE LoRA.

    Process:
    1. Parse arguments and create configs
    2. Load tokenizer and base model
    3. Create dataloaders
    4. Create MoE model with proper initialization
    5. Setup optimizer and scheduler
    6. Training loop with diversity loss
    7. Save best checkpoint
    """
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
    logger.info("MoE LoRA Training Configuration")
    logger.info("=" * 80)
    logger.info("Model Settings:")
    logger.info(f"  Model Name: {training_config.model_name}")
    logger.info(f"  Load in 4-bit: {training_config.load_in_4bit}")
    logger.info(f"  Flash Attention: {training_config.use_flash_attention}")
    logger.info("")
    logger.info("MoE LoRA Settings:")
    logger.info(f"  Number of Experts: {moe_config.num_experts}")
    logger.info(f"  Top-K Experts: {moe_config.top_k}")
    logger.info(f"  LoRA Rank: {moe_config.r}")
    logger.info(f"  LoRA Alpha: {moe_config.lora_alpha}")
    logger.info(f"  LoRA Dropout: {moe_config.lora_dropout}")
    logger.info(f"  Adapter Init: {moe_config.adapter_init}")
    logger.info(f"  Router Init: {moe_config.router_init}")
    logger.info(f"  Target Modules: {moe_config.target_modules}")
    logger.info("")
    logger.info("Training Settings:")
    logger.info(f"  Epochs: {training_config.num_epochs}")
    logger.info(f"  Batch Size: {training_config.batch_size}")
    logger.info(f"  Gradient Accumulation: {training_config.gradient_accumulation_steps}")
    logger.info(f"  Learning Rate: {training_config.learning_rate}")
    logger.info(f"  Diversity Loss Weight: {args.diversity_loss_coefficient}")
    logger.info(f"  Max Length: {training_config.max_length}")
    logger.info(f"  Output Directory: {output_path}")
    logger.info("=" * 80)

    logger.info("Loading tokenizer...")
    tokenizer = get_tokenizer(
        model_name=training_config.model_name,
        trust_remote_code=True,
    )
    logger.info(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

    logger.info("Loading base model...")
    base_model = get_model(
        model_name=training_config.model_name,
        load_in_4bit=training_config.load_in_4bit,
        torch_dtype=torch.bfloat16,
        use_flash_attention=training_config.use_flash_attention,
    )
    logger.info(f"Base model loaded: {base_model.__class__.__name__}")

    logger.info("Creating dataloaders...")
    train_dataloader, val_dataloader = create_dataloaders(
        tokenizer=tokenizer,
        config=training_config,
    )
    logger.info(f"Train batches: {len(train_dataloader)}")
    logger.info(f"Validation batches: {len(val_dataloader)}")

    logger.info("Creating MoE LoRA model...")

    adapter_name = f"{args.adapter_init}_{args.router_init}"

    moe_model = MoELoRAModel(
        model=base_model,
        peft_config=moe_config,
        adapter_name=adapter_name,
    )

    logger.info("MoE LoRA model created successfully")
    moe_model.print_trainable_parameters()

    logger.info("Setting up training parameters...")

    logger.info("Freezing base model parameters...")
    base_param_count = 0
    for param in base_model.parameters():
        param.requires_grad = False
        base_param_count += param.numel()
    logger.info(f"Frozen {base_param_count:,} base model parameters")

    logger.info("Enabling MoE parameters for training...")
    moe_params = moe_model.get_trainable_parameters()
    moe_param_count = 0
    for param in moe_params:
        param.requires_grad = True
        moe_param_count += param.numel()
    logger.info(f"Training {moe_param_count:,} MoE parameters")

    logger.info("Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        moe_params,
        lr=training_config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    logger.info(f"Optimizer: AdamW (lr={training_config.learning_rate})")

    logger.info("Setting up learning rate scheduler...")
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
    logger.info(f"Scheduler: CosineAnnealingLR (total_steps={total_steps})")

    logger.info("=" * 80)
    logger.info("Starting Training")
    logger.info("=" * 80)

    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(training_config.num_epochs):
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Epoch {epoch + 1}/{training_config.num_epochs}")
        logger.info("=" * 80)

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

        logger.info(f"Epoch {epoch + 1} Training Loss: {train_loss:.4f}")

        val_loss = validate(
            moe_model=moe_model,
            val_dataloader=val_dataloader,
            device="cuda",
        )
        logger.info(f"Epoch {epoch + 1} Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            checkpoint_path = (
                output_path / f"{args.adapter_init}_{args.router_init}_top{args.top_k}"
            )
            checkpoint_path.mkdir(exist_ok=True)

            logger.info("=" * 80)
            logger.info(f"New best validation loss: {val_loss:.4f}")
            logger.info(f"Saving checkpoint to: {checkpoint_path}")
            logger.info("=" * 80)

            logger.info("Saving MoE model...")
            moe_model.save_pretrained(str(checkpoint_path))

            logger.info("Saving tokenizer...")
            tokenizer.save_pretrained(str(checkpoint_path))

            logger.info("Saving training state...")
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

            logger.info("Checkpoint saved successfully")

    logger.info("")
    logger.info("=" * 80)
    logger.info("Training Completed!")
    logger.info("=" * 80)
    logger.info(f"Best Validation Loss: {best_val_loss:.4f}")
    logger.info(f"Best Epoch: {best_epoch + 1}/{training_config.num_epochs}")
    logger.info(f"Model saved to: {output_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
