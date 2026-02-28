#!/usr/bin/env python3
"""
Create extracted class files and update class_mapping.json with file references.
"""

import re
import json
import os
import shutil

def extract_class_by_line(filename, start_line):
    """Extract class content starting from a specific line."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Get content from start_line to next class or EOF
        content = []
        for i in range(start_line - 1, len(lines)):
            line = lines[i]
            # Stop if we hit another class definition (but not the first one)
            if i > start_line - 1 and re.match(r'^class\s+\w+', line.strip()):
                break
            content.append(line)

        return ''.join(content)
    except Exception as e:
        print(f"Error: {e}")
        return None

# Create class extracts directory
if os.path.exists('class_extracts_final'):
    shutil.rmtree('class_extracts_final')

os.makedirs('class_extracts_final/qwen3_5_moe', exist_ok=True)
os.makedirs('class_extracts_final/qwen3_next', exist_ok=True)

# Load the current mapping
with open('class_mapping.json', 'r') as f:
    mapping_data = json.load(f)

print("Creating extracted class files and updating mapping...")

# Process core architecture mappings
core_mappings = mapping_data['class_mappings'][0]['mappings']
for item in core_mappings:
    component = item['component']
    file1_class = item['file1_class']
    file1_line = item['file1_line']
    file2_class = item['file2_class']
    file2_line = item['file2_line']

    # Create file names
    file1_extract = f"class_extracts_final/qwen3_5_moe/{file1_class}.py"
    file2_extract = f"class_extracts_final/qwen3_next/{file2_class}.py"

    # Extract both classes
    file1_content = extract_class_by_line('modeling_qwen3_5_moe.py', file1_line)
    file2_content = extract_class_by_line('modeling_qwen3_next.py', file2_line)

    if file1_content:
        with open(file1_extract, 'w') as f:
            f.write(file1_content)
        item['file1_extract'] = file1_extract
        print(f"  ✓ Created: {file1_extract}")

    if file2_content:
        with open(file2_extract, 'w') as f:
            f.write(file2_content)
        item['file2_extract'] = file2_extract
        print(f"  ✓ Created: {file2_extract}")

# Handle Rotary Embeddings special case
print("Processing RotaryEmbedding...")
rotary_mapping = mapping_data['class_mappings'][1]['mappings'][0]
file1_classes = rotary_mapping['file1_classes']
file1_lines = rotary_mapping['file1_lines']
file2_class = rotary_mapping['file2_class']
file2_line = rotary_mapping['file2_line']

# Extract both rotary embedding classes from file1
file1_extracts = []
for i, class_name in enumerate(file1_classes):
    content = extract_class_by_line('modeling_qwen3_5_moe.py', file1_lines[i])
    if content:
        extract_file = f"class_extracts_final/qwen3_5_moe/{class_name}.py"
        with open(extract_file, 'w') as f:
            f.write(content)
        file1_extracts.append(extract_file)
        print(f"  ✓ Created: {extract_file}")

rotary_mapping['file1_extracts'] = file1_extracts

# Extract rotary embedding from file2
file2_content = extract_class_by_line('modeling_qwen3_next.py', file2_line)
if file2_content:
    extract_file = f"class_extracts_final/qwen3_next/{file2_class}.py"
    with open(extract_file, 'w') as f:
        f.write(file2_content)
    rotary_mapping['file2_extract'] = extract_file
    print(f"  ✓ Created: {extract_file}")

# Process unique classes for qwen3_5_moe
print("Processing unique qwen3_5_moe classes...")
unique_moe = mapping_data['class_mappings'][2]['classes']
for item in unique_moe:
    class_name = item['class_name']
    line = item['line']

    content = extract_class_by_line('modeling_qwen3_5_moe.py', line)
    if content:
        extract_file = f"class_extracts_final/qwen3_5_moe/{class_name}.py"
        with open(extract_file, 'w') as f:
            f.write(content)
        item['extract_file'] = extract_file
        print(f"  ✓ Created: {extract_file}")

# Process unique classes for qwen3_next
print("Processing unique qwen3_next classes...")
unique_next = mapping_data['class_mappings'][3]['classes']
for item in unique_next:
    class_name = item['class_name']
    line = item['line']

    content = extract_class_by_line('modeling_qwen3_next.py', line)
    if content:
        extract_file = f"class_extracts_final/qwen3_next/{class_name}.py"
        with open(extract_file, 'w') as f:
            f.write(content)
        item['extract_file'] = extract_file
        print(f"  ✓ Created: {extract_file}")

# Add metadata about the extraction directory
mapping_data['extraction_info'] = {
    'extract_dir': 'class_extracts_final',
    'qwen3_5_moe_dir': 'class_extracts_final/qwen3_5_moe',
    'qwen3_next_dir': 'class_extracts_final/qwen3_next',
    'total_extracted_files': 0,
    'extraction_notes': 'Each class has been extracted to its own Python file for easy comparison'
}

# Count total extracted files
moe_count = len([f for f in os.listdir('class_extracts_final/qwen3_5_moe') if f.endswith('.py')])
next_count = len([f for f in os.listdir('class_extracts_final/qwen3_next') if f.endswith('.py')])
mapping_data['extraction_info']['total_extracted_files'] = moe_count + next_count
mapping_data['extraction_info']['qwen3_5_moe_files'] = moe_count
mapping_data['extraction_info']['qwen3_next_files'] = next_count

# Save the updated mapping
with open('class_mapping_updated.json', 'w') as f:
    json.dump(mapping_data, f, indent=2)

print(f"\n{'='*60}")
print(f"✅ Successfully created {moe_count + next_count} extracted class files")
print(f"📁 Directory: class_extracts_final/")
print(f"📄 Updated mapping: class_mapping_updated.json")
print(f"{'='*60}")

# Create a summary file
summary_content = """# Extracted Class Files

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
diff class_extracts_final/qwen3_next/Qwen3NextAttention.py \\
     class_extracts_final/qwen3_5_moe/Qwen3_5MoeAttention.py
```

## Statistics

- Total extracted files: """ + str(moe_count + next_count) + """
- qwen3_5_moe classes: """ + str(moe_count) + """
- qwen3_next classes: """ + str(next_count) + """

## Updated Mapping

The `class_mapping_updated.json` file now includes references to these extracted files:
- Each mapping has `file1_extract` and/or `file2_extract` fields
- Unique classes have `extract_file` fields
- Extraction metadata is in `extraction_info`

This makes it easy to programmatically access individual class files for comparison or analysis.
"""

with open('class_extracts_final/README.md', 'w') as f:
    f.write(summary_content)

print(f"📄 Created summary: class_extracts_final/README.md")