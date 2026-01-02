from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    model_name: str
    output_dir: str = "./checkpoints"
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_length: int = 512
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    load_in_4bit: bool = True
    use_flash_attention: bool = False
