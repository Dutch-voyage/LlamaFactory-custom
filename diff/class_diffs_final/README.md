# Git Diff Files Summary

This directory contains git diff files comparing mapped classes between:
- `modeling_qwen3_next.py` (Qwen3 Next) → `modeling_qwen3_5_moe.py` (Qwen3.5 MoE + Vision)

## Files Created

**Note**: Diffs show changes FROM qwen3_next TO qwen3_5_moe (reversed order)

- **DynamicCache.diff**: `Qwen3NextDynamicCache` → `Qwen3_5MoeDynamicCache`
- **RMSNormGated.diff**: `Qwen3NextRMSNormGated` → `Qwen3_5MoeRMSNormGated`
- **RMSNorm.diff**: `Qwen3NextRMSNorm` → `Qwen3_5MoeRMSNorm`
- **GatedDeltaNet.diff**: `Qwen3NextGatedDeltaNet` → `Qwen3_5MoeGatedDeltaNet`
- **Attention.diff**: `Qwen3NextAttention` → `Qwen3_5MoeAttention`
- **MLP.diff**: `Qwen3NextMLP` → `Qwen3_5MoeMLP`
- **Experts.diff**: `Qwen3NextExperts` → `Qwen3_5MoeExperts`
- **TopKRouter.diff**: `Qwen3NextTopKRouter` → `Qwen3_5MoeTopKRouter`
- **SparseMoeBlock.diff**: `Qwen3NextSparseMoeBlock` → `Qwen3_5MoeSparseMoeBlock`
- **DecoderLayer.diff**: `Qwen3NextDecoderLayer` → `Qwen3_5MoeDecoderLayer`
- **PreTrainedModel.diff**: `Qwen3NextPreTrainedModel` → `Qwen3_5MoePreTrainedModel`
- **Model.diff**: `Qwen3NextModel` → `Qwen3_5MoeModel`
- **ForCausalLM.diff**: `Qwen3NextForCausalLM` → `Qwen3_5MoeForCausalLM`
- **RotaryEmbedding.diff**: Single rotary embedding → Combined vision+text rotary embeddings

## Usage

View individual diffs:
```bash
cat class_diffs_final/<component>.diff
```

Example:
```bash
cat class_diffs_final/Attention.diff
```

## Diff Direction

All diffs show changes **FROM** `qwen3_next` **TO** `qwen3_5_moe`:
- Lines starting with `+` are additions in qwen3_5_moe
- Lines starting with `-` are deletions from qwen3_next

This means you can see what features/additions qwen3_5_moe has compared to qwen3_next.

## Statistics

- Total mappings processed: 14
- Failed mappings: 0
- Success rate: 100.0%
