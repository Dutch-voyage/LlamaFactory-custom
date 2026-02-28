#!/usr/bin/env python3
"""
Extract Python classes according to class_mapping.json and create git diffs.
"""

import re
import subprocess
import json
import os

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

# Clean up previous runs
import shutil
if os.path.exists('class_diffs_final'):
    shutil.rmtree('class_diffs_final')
os.makedirs('class_diffs_final', exist_ok=True)

# Load the mapping
with open('class_mapping.json', 'r') as f:
    mapping_data = json.load(f)

print("Extracting classes and creating git diffs (qwen3_next → qwen3_5_moe)...\n")

# Track success
successful_mappings = 0
failed_mappings = 0

# Process core architecture mappings
core_mappings = mapping_data['class_mappings'][0]['mappings']
for item in core_mappings:
    component = item['component']
    file1_class = item['file1_class']
    file1_line = item['file1_line']
    file2_class = item['file2_class']
    file2_line = item['file2_line']

    print(f"Processing {component}:")
    print(f"  {file2_class} (line {file2_line}) → {file1_class} (line {file1_line})")

    # Extract both classes
    file1_content = extract_class_by_line('modeling_qwen3_5_moe.py', file1_line)
    file2_content = extract_class_by_line('modeling_qwen3_next.py', file2_line)

    if file1_content and file2_content:
        # Write to temp files
        temp_file1 = f'temp_{file1_class}.py'
        temp_file2 = f'temp_{file2_class}.py'

        with open(temp_file1, 'w') as f:
            f.write(file1_content)
        with open(temp_file2, 'w') as f:
            f.write(file2_content)

        # Create git diff using simple command (reversed order: qwen3_next -> qwen3_5_moe)
        diff_file = f'class_diffs_final/{component}.diff'

        # Use simple git diff --no-index command (reversed order)
        with open(diff_file, 'w') as outfile:
            result = subprocess.run([
                'git', 'diff', '--no-index',
                temp_file2, temp_file1  # Reversed: qwen3_next first, qwen3_5_moe second
            ], stdout=outfile, stderr=subprocess.DEVNULL)

        # Get stats
        with open(diff_file, 'r') as f:
            diff_content = f.read()
            additions = diff_content.count('+') - diff_content.count('+++')
            deletions = diff_content.count('-') - diff_content.count('---')

        print(f"  ✓ Created {component}.diff ({additions} additions, {deletions} deletions)")
        successful_mappings += 1

        # Clean up temp files
        os.remove(temp_file1)
        os.remove(temp_file2)
    else:
        print(f"  ✗ Failed to extract content")
        failed_mappings += 1
    print()

# Handle Rotary Embeddings special case
print("Processing RotaryEmbedding (special case):")
rotary_mapping = mapping_data['class_mappings'][1]['mappings'][0]
file1_classes = rotary_mapping['file1_classes']
file1_lines = rotary_mapping['file1_lines']
file2_class = rotary_mapping['file2_class']
file2_line = rotary_mapping['file2_line']

# Combine both rotary embedding classes from file1
combined_content = ""
for i, class_name in enumerate(file1_classes):
    content = extract_class_by_line('modeling_qwen3_5_moe.py', file1_lines[i])
    if content:
        combined_content += f"# {class_name}\n"
        combined_content += content + "\n"

file2_content = extract_class_by_line('modeling_qwen3_next.py', file2_line)

if combined_content and file2_content:
    temp_file1 = 'temp_combined_rotary.py'
    temp_file2 = f'temp_{file2_class}.py'

    with open(temp_file1, 'w') as f:
        f.write(combined_content)
    with open(temp_file2, 'w') as f:
        f.write(file2_content)

    diff_file = 'class_diffs_final/RotaryEmbedding.diff'

    with open(diff_file, 'w') as outfile:
        result = subprocess.run([
            'git', 'diff', '--no-index',
            temp_file2, temp_file1  # Reversed: qwen3_next first, qwen3_5_moe second
        ], stdout=outfile, stderr=subprocess.DEVNULL)

    # Get stats
    with open(diff_file, 'r') as f:
        diff_content = f.read()
        additions = diff_content.count('+') - diff_content.count('+++')
        deletions = diff_content.count('-') - diff_content.count('---')

    print(f"  ✓ Created RotaryEmbedding.diff ({additions} additions, {deletions} deletions)")
    print(f"    {file2_class} → Combined: {file1_classes[0]} + {file1_classes[1]}")
    successful_mappings += 1

    os.remove(temp_file1)
    os.remove(temp_file2)
else:
    print(f"  ✗ Failed to extract content")
    failed_mappings += 1

print(f"\n{'='*60}")
print(f"✅ Successfully created {successful_mappings} git diff files")
print(f"📁 Directory: class_diffs_final/")
print(f"{'='*60}")

# Create a summary file
summary_content = """# Git Diff Files Summary

This directory contains git diff files comparing mapped classes between:
- `modeling_qwen3_next.py` (Qwen3 Next) → `modeling_qwen3_5_moe.py` (Qwen3.5 MoE + Vision)

## Files Created

**Note**: Diffs show changes FROM qwen3_next TO qwen3_5_moe (reversed order)

"""
for item in core_mappings:
    component = item['component']
    file1_class = item['file2_class']  # Reversed order
    file2_class = item['file1_class']  # Reversed order
    summary_content += f"- **{component}.diff**: `{file1_class}` → `{file2_class}`\n"

summary_content += f"- **RotaryEmbedding.diff**: Single rotary embedding → Combined vision+text rotary embeddings\n"

summary_content += f"""
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

- Total mappings processed: {successful_mappings}
- Failed mappings: {failed_mappings}
- Success rate: {successful_mappings/(successful_mappings+failed_mappings)*100:.1f}%
"""

with open('class_diffs_final/README.md', 'w') as f:
    f.write(summary_content)

print(f"📄 Created summary: class_diffs_final/README.md")