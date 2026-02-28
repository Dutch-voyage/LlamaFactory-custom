class Qwen3NextForQuestionAnswering(GenericForQuestionAnswering, Qwen3NextPreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`


__all__ = [
    "Qwen3NextForCausalLM",
    "Qwen3NextForQuestionAnswering",
    "Qwen3NextModel",
    "Qwen3NextPreTrainedModel",
    "Qwen3NextForSequenceClassification",
    "Qwen3NextForTokenClassification",
]
