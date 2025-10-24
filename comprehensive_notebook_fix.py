#!/usr/bin/env python3
"""
Comprehensive fix for notebook formatting issues
"""

import json
import re
import os

def fix_compressed_code(text):
    """Fix compressed code by adding proper line breaks"""
    # Add line breaks after imports
    text = re.sub(r'(import\s+\w+)', r'\1\n', text)
    text = re.sub(r'(from\s+\w+\s+import)', r'\1\n', text)
    
    # Add line breaks after function definitions
    text = re.sub(r'(def\s+\w+\([^)]*\):)', r'\1\n', text)
    
    # Add line breaks after class definitions
    text = re.sub(r'(class\s+\w+[^:]*:)', r'\1\n', text)
    
    # Add line breaks after control structures
    text = re.sub(r'(if\s+[^:]+:)', r'\1\n', text)
    text = re.sub(r'(for\s+[^:]+:)', r'\1\n', text)
    text = re.sub(r'(while\s+[^:]+:)', r'\1\n', text)
    text = re.sub(r'(else:)', r'\1\n', text)
    text = re.sub(r'(elif\s+[^:]+:)', r'\1\n', text)
    text = re.sub(r'(try:)', r'\1\n', text)
    text = re.sub(r'(except[^:]*:)', r'\1\n', text)
    text = re.sub(r'(finally:)', r'\1\n', text)
    
    # Add line breaks after print statements
    text = re.sub(r'(print\([^)]*\))', r'\1\n', text)
    
    # Add line breaks after assignments
    text = re.sub(r'(\w+\s*=\s*[^=\n]+)(?=\w+\s*=)', r'\1\n', text)
    
    # Add line breaks after comments
    text = re.sub(r'(#\s+[A-Z][^#\n]*)', r'\n\1\n', text)
    
    # Add line breaks after method calls
    text = re.sub(r'(\w+\([^)]*\))(?=\w+\()', r'\1\n', text)
    
    # Clean up multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def fix_markdown_formatting(text):
    """Fix markdown formatting"""
    # Add line breaks after headers
    text = re.sub(r'(##\s+[A-Z][^\n]*)', r'\1\n', text)
    text = re.sub(r'(###\s+[A-Z][^\n]*)', r'\1\n', text)
    text = re.sub(r'(####\s+[A-Z][^\n]*)', r'\1\n', text)
    
    # Add line breaks after task lists
    text = re.sub(r'(\d+\.\s+[A-Z][^\n]*)', r'\1\n', text)
    
    # Add line breaks after bold text
    text = re.sub(r'(\*\*[^*]+\*\*)', r'\1\n', text)
    
    # Clean up multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def fix_notebook_cell(cell):
    """Fix a single notebook cell"""
    if cell['cell_type'] == 'code' and 'source' in cell:
        if isinstance(cell['source'], list):
            # Join all lines
            full_text = ''.join(cell['source'])
            
            # Fix compressed code
            fixed_text = fix_compressed_code(full_text)
            
            # Split into proper lines
            lines = fixed_text.split('\n')
            cleaned_lines = []
            
            for line in lines:
                line = line.strip()
                if line:
                    cleaned_lines.append(line)
            
            cell['source'] = cleaned_lines
    
    elif cell['cell_type'] == 'markdown' and 'source' in cell:
        if isinstance(cell['source'], list):
            # Join all lines
            full_text = ''.join(cell['source'])
            
            # Fix markdown formatting
            fixed_text = fix_markdown_formatting(full_text)
            
            # Split into proper lines
            lines = fixed_text.split('\n')
            cleaned_lines = []
            
            for line in lines:
                line = line.strip()
                if line:
                    cleaned_lines.append(line)
            
            cell['source'] = cleaned_lines
    
    return cell

def fix_notebook(notebook_path):
    """Fix formatting in a single notebook"""
    print(f"Fixing {notebook_path}...")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    changes_made = False
    
    for cell in notebook['cells']:
        original_source = cell.get('source', [])
        fixed_cell = fix_notebook_cell(cell)
        
        if fixed_cell.get('source', []) != original_source:
            changes_made = True
    
    if changes_made:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print(f"  Fixed {notebook_path}")
    else:
        print(f"  No changes needed for {notebook_path}")

def main():
    """Fix all notebooks"""
    print("Comprehensive notebook formatting fix...")
    print("=" * 60)
    
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
    
    print("=" * 60)
    print("Comprehensive formatting fix completed!")

if __name__ == "__main__":
    main()
