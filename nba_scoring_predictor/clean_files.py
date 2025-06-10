
# -*- coding: utf-8 -*-
"""
Script to clean files and ensure proper UTF-8 encoding
"""
import os
import re

def clean_file_encoding(file_path):
    """Clean a file to ensure proper UTF-8 encoding."""
    try:
        # Read file with various encodings
        content = None
        encodings = ['utf-8', 'latin1', 'cp1252', 'ascii']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                print(f"Successfully read {file_path} with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            print(f"Could not read {file_path} with any encoding")
            return False
        
        # Replace problematic characters
        replacements = {
            'R': 'R-squared',
            '²': '-squared',
            ''': "'",
            ''': "'",
            '"': '"',
            '"': '"',
            '–': '-',
            '—': '-',
            '…': '...'
        }
        
        for old, new in replacements.items():
            content = content.replace(old, new)
        
        # Ensure file starts with UTF-8 encoding declaration
        if not content.startswith('# -*- coding: utf-8 -*-'):
            if content.startswith('#'):
                # Replace existing encoding declaration
                lines = content.split('\n')
                if 'coding' in lines[0]:
                    lines[0] = '# -*- coding: utf-8 -*-'
                else:
                    lines.insert(0, '# -*- coding: utf-8 -*-')
                content = '\n'.join(lines)
            else:
                content = '# -*- coding: utf-8 -*-\n' + content
        
        # Write file back with UTF-8 encoding
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Cleaned and saved {file_path}")
        return True
        
    except Exception as e:
        print(f"Error cleaning {file_path}: {e}")
        return False

def clean_all_python_files():
    """Clean all Python files in the project."""
    # Get project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk(project_root):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"Found {len(python_files)} Python files to clean:")
    
    cleaned_count = 0
    for file_path in python_files:
        print(f"Cleaning: {file_path}")
        if clean_file_encoding(file_path):
            cleaned_count += 1
    
    print(f"\nCleaned {cleaned_count}/{len(python_files)} files successfully")

if __name__ == "__main__":
    clean_all_python_files()