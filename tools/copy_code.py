#!/usr/bin/env python3
"""
copy_code.py - Copies the full directory tree and all files to clipboard with file names.

This script recursively traverses the current directory (or a specified directory),
creates a formatted representation of the file tree structure with file contents,
and copies it to the system clipboard.
"""

import os
import sys
import pyperclip
from pathlib import Path
from typing import List, Optional


def get_file_tree(
    directory: str = ".", 
    ignore_patterns: Optional[List[str]] = None,
    max_file_size: int = 1024 * 1024  # 1MB default limit
) -> str:
    """
    Generate a formatted representation of the directory tree with file contents.
    
    Args:
        directory: The directory to scan (default: current directory)
        ignore_patterns: List of patterns to ignore (default: None)
        max_file_size: Maximum file size to include in bytes (default: 1MB)
    
    Returns:
        Formatted string representation of the directory tree
    """
    if ignore_patterns is None:
        ignore_patterns = [
            "__pycache__",
            ".git",
            ".gitignore",
            ".env",
            ".env.local",
            "node_modules",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            ".DS_Store",
            "Thumbs.db",
            "*.log",
            "*.tmp",
            "copy_code.py",  # Don't include this script itself
            "*.png",  # Image files
            "*.jpg",
            "*.jpeg",
            "*.gif",
            "*.pdf",
            "*.pkl",  # Model files
            "test_*.py",  # Test files
            "*_test.py",
        ]
    
    result = []
    root_path = Path(directory).resolve()
    
    # Add header
    result.append(f"DIRECTORY TREE: {root_path}")
    result.append("=" * 60)
    result.append("")
    
    def should_ignore(path: Path) -> bool:
        """Check if a path should be ignored based on patterns."""
        for pattern in ignore_patterns:
            if pattern.startswith("*."):
                # File extension pattern
                if path.suffix == pattern[1:]:
                    return True
            elif pattern in path.parts or path.name == pattern:
                return True
        return False
    
    def process_directory(current_path: Path, prefix: str = ""):
        """Recursively process directories and files."""
        # Get all items and sort them
        try:
            items = sorted(current_path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            result.append(f"{prefix}[Permission Denied: {current_path.name}]")
            return
        
        for i, item in enumerate(items):
            # Skip ignored items
            if should_ignore(item):
                continue
                
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            tree_prefix = "    " if is_last else "│   "
            
            if item.is_dir():
                result.append(f"{prefix}{current_prefix}[DIR] {item.name}/")
                process_directory(item, prefix + tree_prefix)
            else:
                result.append(f"{prefix}{current_prefix}{item.name}")
                
                # Try to read and add file content
                try:
                    if item.stat().st_size <= max_file_size:
                        with open(item, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            if content.strip():  # Only add non-empty files
                                result.append(f"{prefix}    ── Content ──")
                                # Add content with proper indentation
                                for line in content.splitlines():
                                    # Replace special characters that might cause encoding issues
                                    safe_line = line.replace('✓', '[OK]').replace('└', '`--').replace('├', '|--').replace('│', '|')
                                    result.append(f"{prefix}    {safe_line}")
                                result.append(f"{prefix}    ── End Content ──")
                    else:
                        result.append(f"{prefix}    [File too large: {item.stat().st_size} bytes]")
                except Exception as e:
                    result.append(f"{prefix}    [Error reading file: {e}]")
                
                result.append("")
    
    process_directory(root_path)
    return "\n".join(result)


def main():
    """Main function to execute the script."""
    try:
        # Get directory from command line argument or use current directory
        directory = sys.argv[1] if len(sys.argv) > 1 else "."
        
        print(f"Scanning directory: {os.path.abspath(directory)}")
        print("This may take a moment for large directories...")
        
        # Generate the file tree representation
        file_tree = get_file_tree(directory)
        
        # Copy to clipboard
        pyperclip.copy(file_tree)
        
        print(f"\nSuccessfully copied directory tree to clipboard!")
        print(f"Total characters copied: {len(file_tree)}")
        
        # Show a preview (safe for Windows encoding)
        lines = file_tree.split('\n')[:20]
        print(f"\nPreview of copied content:")
        for line in lines:
            # Encode to avoid Windows encoding issues
            safe_line = line.encode('ascii', 'ignore').decode('ascii')
            print(safe_line)
        if len(file_tree.split('\n')) > 20:
            print("... (content truncated)")
            
    except ImportError:
        print("Error: pyperclip module not found.")
        print("Install it with: pip install pyperclip")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()