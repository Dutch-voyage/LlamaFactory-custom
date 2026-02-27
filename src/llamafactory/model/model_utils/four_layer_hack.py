# Copyright 2025 the LlamaFactory team.
#
# Temporary hack to load only the last 4 layers of a model
# This is useful for testing with large models when memory is limited

from typing import Any, Dict

from ..extras import logging


logger = logging.get_logger(__name__)


def apply_four_layer_hack(
    load_class,
    model_args: Any,
    init_kwargs: Dict[str, Any],
    config: Any,
) -> bool:
    """
    Apply a temporary hack to load only the last 4 layers of a model.

    This function modifies the config to reduce the number of layers to 4,
    then loads only the last 4 layers' weights from the checkpoint.

    Args:
        load_class: The AutoModel class to use for loading
        model_args: Model arguments containing the model path
        init_kwargs: Initialization kwargs for loading the model
        config: The model configuration (will be modified)

    Returns:
        bool: True if the hack was applied, False otherwise
    """
    original_num_layers = getattr(config, "num_hidden_layers", getattr(config, "num_layers", None))

    if not original_num_layers or original_num_layers <= 4:
        return False

    logger.warning_rank0(f"HACK: Reducing model from {original_num_layers} layers to 4 layers (last 4)")

    # Store original layer count for loading weights
    init_kwargs["_original_num_layers"] = original_num_layers

    # Modify config to only have 4 layers
    setattr(config, "num_hidden_layers", 4)
    if hasattr(config, "num_layers"):
        setattr(config, "num_layers", 4)

    # Load model with reduced config
    model = load_class.from_pretrained(**init_kwargs)

    # Handle special model types
    if getattr(model.config, "model_type", None) in ["qwen2_5_omni", "qwen3_omni_moe"]:
        model = getattr(model, "thinker")

    # Now manually load only the last 4 layers from checkpoint
    checkpoint_path = model_args.model_name_or_path

    # Load the full model checkpoint to get state dict
    full_model_kwargs = {k: v for k, v in init_kwargs.items()
                        if k not in ["_original_num_layers", "config", "pretrained_model_name_or_path"]}
    full_model = load_class.from_pretrained(
        pretrained_model_name_or_path=checkpoint_path,
        config=config,
        **full_model_kwargs
    )
    state_dict = full_model.state_dict()

    # Filter to only load last 4 layers
    filtered_state_dict = {}
    for key, value in state_dict.items():
        # Check if this key contains layer references
        if ".layers." in key or ".layer." in key:
            # Extract layer number and only keep last 4 layers
            parts = key.split(".")
            for part in parts:
                if part.isdigit():
                    layer_num = int(part)
                    if layer_num >= original_num_layers - 4:
                        # Remap to 0-3 range
                        new_layer_num = layer_num - (original_num_layers - 4)
                        new_parts = [str(new_layer_num) if p.isdigit() else p for p in parts]
                        new_key = ".".join(new_parts)
                        filtered_state_dict[new_key] = value
                    break
        else:
            # Keep all non-layer parameters (embeddings, layernorm, etc.)
            filtered_state_dict[key] = value

    # Load the filtered state dict
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    logger.info_rank0(f"HACK: Loaded last 4 layers. Missing keys: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")

    return True