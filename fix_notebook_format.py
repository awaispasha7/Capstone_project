#!/usr/bin/env python3
"""
Fix notebook formatting issues
"""

import json
import re
import os

def fix_cell_formatting(cell):
    """Fix formatting issues in a cell"""
    if cell['cell_type'] == 'code' and 'source' in cell:
        # Fix compressed code by adding proper line breaks
        source = cell['source']
        if isinstance(source, list):
            # Join all lines and then split by common patterns
            full_text = ''.join(source)
            
            # Add line breaks after common Python patterns
            full_text = re.sub(r'(import\s+\w+)', r'\1\n', full_text)
            full_text = re.sub(r'(from\s+\w+\s+import)', r'\1\n', full_text)
            full_text = re.sub(r'(def\s+\w+)', r'\1\n', full_text)
            full_text = re.sub(r'(class\s+\w+)', r'\1\n', full_text)
            full_text = re.sub(r'(if\s+.*:)', r'\1\n', full_text)
            full_text = re.sub(r'(for\s+.*:)', r'\1\n', full_text)
            full_text = re.sub(r'(while\s+.*:)', r'\1\n', full_text)
            full_text = re.sub(r'(else:)', r'\1\n', full_text)
            full_text = re.sub(r'(elif\s+.*:)', r'\1\n', full_text)
            full_text = re.sub(r'(print\()', r'\1\n', full_text)
            full_text = re.sub(r'(#\s+[A-Z])', r'\n\1', full_text)
            
            # Split into lines and clean up
            lines = full_text.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line:
                    cleaned_lines.append(line)
            
            cell['source'] = cleaned_lines
    
    elif cell['cell_type'] == 'markdown' and 'source' in cell:
        # Fix markdown formatting
        source = cell['source']
        if isinstance(source, list):
            full_text = ''.join(source)
            
            # Add proper line breaks for markdown
            full_text = re.sub(r'(##\s+[A-Z])', r'\n\1', full_text)
            full_text = re.sub(r'(###\s+[A-Z])', r'\n\1', full_text)
            full_text = re.sub(r'(\d+\.\s+[A-Z])', r'\n\1', full_text)
            full_text = re.sub(r'(\*\*[A-Z])', r'\n\1', full_text)
            
            # Split into lines
            lines = full_text.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line:
                    cleaned_lines.append(line)
            
            cell['source'] = cleaned_lines
    
    return cell

def fix_notebook(notebook_path):
    """Fix formatting in a single notebook"""
    print(f"Fixing formatting in {notebook_path}...")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    changes_made = False
    
    for cell in notebook['cells']:
        original_cell = cell.copy()
        fixed_cell = fix_cell_formatting(cell)
        
        if fixed_cell != original_cell:
            changes_made = True
    
    if changes_made:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print(f"  Fixed formatting in {notebook_path}")
    else:
        print(f"  No formatting fixes needed for {notebook_path}")

def main():
    """Fix formatting in all notebooks"""
    print("Fixing notebook formatting...")
    print("=" * 50)
    
    notebooks_dir = "notebooks"
    
    # List of all question notebooks
    question_files = [
        "Question 1.ipynb",
        "Question 2.ipynb",
        "Question 3.ipynb",
        "Question 4.ipynb",
        "Question 5.ipynb",
        "Question 6.ipynb",
        "Question 7.ipynb",
        "Question 8.ipynb",
        "Question 9.ipynb"
    ]
    
    for question_file in question_files:
        notebook_path = os.path.join(notebooks_dir, question_file)
        if os.path.exists(notebook_path):
            fix_notebook(notebook_path)
        else:
            print(f"  Not found: {notebook_path}")
    
    print("=" * 50)
    print("Formatting fixes completed!")

if __name__ == "__main__":
    main()
