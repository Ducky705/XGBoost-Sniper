import os
import pyperclip

def generate_tree(path, prefix="", is_last=True):
    """Generate a tree structure string"""
    name = os.path.basename(path)
    tree_str = f"{prefix}{'└── ' if is_last else '├── '}{name}\n"
    
    if os.path.isdir(path):
        try:
            items = sorted(os.listdir(path))
            # Separate files and directories
            dirs = [item for item in items if os.path.isdir(os.path.join(path, item))]
            files = [item for item in items if os.path.isfile(os.path.join(path, item))]
            
            # Process directories first
            for i, item in enumerate(dirs):
                item_path = os.path.join(path, item)
                extension = "    " if is_last else "│   "
                tree_str += generate_tree(item_path, prefix + extension, i == len(dirs) - 1)
            
            # Process files
            for i, item in enumerate(files):
                item_path = os.path.join(path, item)
                extension = "    " if is_last else "│   "
                is_file_last = i == len(files) - 1 and len(dirs) == 0
                tree_str += f"{prefix + extension}{'└── ' if is_file_last else '├── '}{item}\n"
                
        except PermissionError:
            tree_str += f"{prefix}    [Permission Denied]\n"
    
    return tree_str

# Generate tree for current directory
current_dir = os.getcwd()
tree_output = f"{os.path.basename(current_dir)}\n"
tree_output += generate_tree(current_dir)

# Copy to clipboard
pyperclip.copy(tree_output)
print("File tree copied to clipboard successfully!")
print("\nTree structure:")
print(tree_output)