class Qwen3NextPreTrainedModel(PreTrainedModel):
    config: Qwen3NextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3NextDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*"]
    _can_record_outputs = {
        "router_logits": OutputRecorder(Qwen3NextTopKRouter, index=0),
        "hidden_states": Qwen3NextDecoderLayer,
        "attentions": Qwen3NextAttention,
    }
    _is_stateful = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Qwen3NextGatedDeltaNet):
            init.ones_(module.dt_bias)
            init.copy_(module.A_log, torch.empty_like(module.A_log).uniform_(0, 16).log_())
        # We initialize with 0s to be 1 centered as the RMSNorm here does (1 + weight)
        elif isinstance(module, Qwen3NextRMSNorm):
            init.zeros_(module.weight)
        elif isinstance(module, Qwen3NextExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, Qwen3NextSparseMoeBlock):
            init.normal_(module.gate.weight, mean=0.0, std=self.config.initializer_range)


