# Copyright 2025 the LlamaFactory team.
#
# Temporary hack to load only the last 4 layers of a model
# This is useful for testing with large models when memory is limited

from typing import Any, Dict
from transformers import PretrainedConfig
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig


from ...extras import logging


logger = logging.get_logger(__name__)

# NOTE supporting Qwen3.5 only, with config.text_config and config.vision_config
def apply_four_layer_hack(
    load_class,
    model_args: Any,
    init_kwargs: Dict[str, Any],
    config: PretrainedConfig,
) -> bool:
    """
    Apply a temporary hack to load only the last 4 layers of a model.

    This function modifies the config to reduce the number of layers to 4.
    The model will be loaded with only 4 layers initialized.

    Args:
        load_class: The AutoModel class to use for loading
        model_args: Model arguments containing the model path
        init_kwargs: Initialization kwargs for loading the model
        config: The model configuration (will be modified)

    Returns:
        bool: True if the hack was applied, False otherwise
    """
    original_num_layers = getattr(config.text_config, "num_hidden_layers", getattr(config.text_config, "num_layers", None))
    
    if not original_num_layers or original_num_layers <= 4:
        return False

    logger.warning_rank0(f"HACK: Reducing model from {original_num_layers} layers to 4 layers")

    # Modify config to only have 4 layers
    setattr(config, "num_hidden_layers", 4)
    if hasattr(config, "num_layers"):
        setattr(config, "num_layers", 4)

    # Also handle layer_types if present (Qwen3.5-MoE uses this to determine attention type per layer)
    # If we don't truncate this, Qwen3_5MoeDynamicCache will have mismatched lengths
    if hasattr(config, "layer_types") and isinstance(config.layer_types, list):
        if len(config.layer_types) > 4:
            setattr(config, "layer_types", config.layer_types[-4:])
            logger.warning_rank0(f"HACK: Truncated layer_types from {original_num_layers} to 4 (last 4)")

    # Handle nested text_config (used in multimodal models like Qwen3.5-MoE)
    if hasattr(config, "text_config") and config.text_config:
        text_config = config.text_config
        if hasattr(text_config, "num_hidden_layers"):
            text_config.num_hidden_layers = 4
        if hasattr(text_config, "layer_types") and isinstance(text_config.layer_types, list):
            if len(text_config.layer_types) > 4:
                text_config.layer_types = text_config.layer_types[-4:]
                logger.warning_rank0(f"HACK: Truncated text_config.layer_types from {original_num_layers} to 4 (last 4)")

    # The model will be loaded with the modified config
    return True