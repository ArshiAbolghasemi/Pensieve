import torch
from peft import prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def get_model(*, model_name: str, **kwargs) -> PreTrainedModel:
    """Load a pretrained model with optional quantization.

    Args:
        model_name: HuggingFace model identifier
        load_in_8bit: Load model in 8-bit quantization
        load_in_4bit: Load model in 4-bit quantization
        device_map: Device mapping strategy
        trust_remote_code: Whether to trust remote code
        torch_dtype: Data type for model weights
        use_flash_attention: Whether to use flash attention (if supported)

    Returns:
        model: Loaded model instance

    """
    load_in_4bit = kwargs.get("load_in_4bit", False)
    load_in_8bit = kwargs.get("load_in_8bit", False)

    if load_in_8bit and load_in_4bit:
        msg = "Choose one approach for quantization"
        raise ValueError(msg)

    dtype = kwargs.get("torch_dtype", torch.bfloat16)

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    trust_remote_code = kwargs.get("trust_remote_code", True)
    device_map = kwargs.get("device_map", "auto")

    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "device_map": device_map,
        "dtype": dtype,
    }

    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config

    model_kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if load_in_8bit or load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    model.gradient_checkpointing_enable()

    return model


def get_tokenizer(*, model_name: str, **kwargs) -> PreTrainedTokenizer:
    """Load tokenizer for the model.

    Args:
        model_name: HuggingFace model identifier
        padding_side: Which side to pad on ('left' or 'right')
        trust_remote_code: Whether to trust remote code
        add_eos_token: Whether to add EOS token
        add_bos_token: Whether to add BOS token

    Returns:
        tokenizer: Loaded tokenizer instance

    """
    trust_remote_code = kwargs.get("trust_remote_code", True)
    padding_side = kwargs.get("padding_side", "right")
    add_eos_token = kwargs.get("add_eos_token", True)
    add_bos_token = kwargs.get("add_bos_token", False)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        padding_side=padding_side,
        add_eos_token=add_eos_token,
        add_bos_token=add_bos_token,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer
