# Copy Code Script

This script copies the full directory tree and all files to your clipboard with file names and contents.

## Usage

```bash
# Copy current directory (two ways to run)
python tools/copy_code.py
python tools/copy_code

# Copy a specific directory
python tools/copy_code.py /path/to/directory
python tools/copy_code /path/to/directory
```

## Features

- Recursively scans all files and directories
- Displays file tree structure with proper indentation
- Includes file contents for text files (up to 1MB)
- Automatically ignores common files and directories (`.git`, `node_modules`, etc.)
- Copies everything to clipboard for easy sharing
- Safe encoding for Windows systems

## Ignored Files

The script automatically ignores:
- `.git` directories
- `node_modules` directories
- Python cache files (`__pycache__`, `*.pyc`, etc.)
- System files (`.DS_Store`, `Thumbs.db`)
- Environment files (`.env`, `.env.local`)
- Log and temporary files
- The script itself (`copy_code.py`)

## Requirements

- Python 3.x
- `pyperclip` library: `pip install pyperclip`

## Output Format

The output includes:
1. Directory tree structure with file names
2. File contents for text files (properly indented)
3. File size warnings for large files
4. Error messages for files that can't be read

## Tips

- Large directories may take time to process
- Files larger than 1MB are skipped to avoid memory issues
- Binary files are read with UTF-8 encoding and errors are ignored
- The script is safe to run and doesn't modify any files