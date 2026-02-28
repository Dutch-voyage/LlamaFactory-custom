# Extracted Class Files

This directory contains individual Python class files extracted from:
- `modeling_qwen3_5_moe.py` → `qwen3_5_moe/` directory
- `modeling_qwen3_next.py` → `qwen3_next/` directory

## Directory Structure

```
class_extracts_final/
├── qwen3_5_moe/           # Classes from modeling_qwen3_5_moe.py
│   ├── Qwen3_5MoeAttention.py
│   ├── Qwen3_5MoeMLP.py
│   └── ...
└── qwen3_next/            # Classes from modeling_qwen3_next.py
    ├── Qwen3NextAttention.py
    ├── Qwen3NextMLP.py
    └── ...
```

## Usage

View individual extracted classes:
```bash
cat class_extracts_final/qwen3_5_moe/Qwen3_5MoeAttention.py
cat class_extracts_final/qwen3_next/Qwen3NextAttention.py
```

Compare two classes directly:
```bash
diff class_extracts_final/qwen3_next/Qwen3NextAttention.py \
     class_extracts_final/qwen3_5_moe/Qwen3_5MoeAttention.py
```

## Statistics

- Total extracted files: 42
- qwen3_5_moe classes: 25
- qwen3_next classes: 17

## Updated Mapping

The `class_mapping_updated.json` file now includes references to these extracted files:
- Each mapping has `file1_extract` and/or `file2_extract` fields
- Unique classes have `extract_file` fields
- Extraction metadata is in `extraction_info`

This makes it easy to programmatically access individual class files for comparison or analysis.
