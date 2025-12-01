import os

def write_file(path, content):
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"📝 Updated: {path}")

def main():
    print("🚀 FIXING .GITIGNORE & GITHUB ACTIONS PERMISSIONS...\n")

    # ==============================================================================
    # 1. UPDATE .gitignore
    # ==============================================================================
    gitignore_path = ".gitignore"
    gitignore_content = """# Environment variables and secrets
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Jupyter Notebook
.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Runtime data
pids
*.pid
*.seed
*.pid.lock

# Coverage directory used by tools like istanbul
coverage/

# Dependency directories
node_modules/
jspm_packages/

# Optional npm cache directory
.npm

# Optional eslint cache
.eslintcache

# Output of 'npm pack'
*.tgz

# Yarn Integrity file
.yarn-integrity

# parcel-bundler cache (https://parceljs.org/)
.cache
.parcel-cache

# next.js build output
.next

# nuxt.js build output
.nuxt

# vuepress build output
.vuepress/dist

# Serverless directories
.serverless

# FuseBox cache
.fusebox/

# DynamoDB Local files
.dynamodb/

# TernJS port file
.tern-port

# Stores VSCode versions used for testing VSCode extensions
.vscode-test

# Model files (if they're too large)
# Ignore all .pkl files...
*.pkl
# ...but DO NOT ignore these specific model files.
!models/v1_pyrite.pkl
!models/v2_diamond.pkl

*.joblib
*.h5
*.pb
*.onnx

# Temporary files
temp_tree.txt
*.tmp
*.temp
"""
    write_file(gitignore_path, gitignore_content)

    # ==============================================================================
    # 2. UPDATE GITHUB ACTIONS WORKFLOW (for write permissions)
    # ==============================================================================
    workflow_path = ".github/workflows/daily_update.yml"
    workflow_content = """name: Daily Sniper Update

on:
  schedule:
    - cron: '0 9 * * *'
  workflow_dispatch:

jobs:
  update-stats:
    runs-on: ubuntu-latest
    # ADD WRITE PERMISSIONS FOR PUSHING CHANGES
    permissions:
      contents: write
    
    steps:
      - name: Checkout Repo
        uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install system dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y libfreetype6-dev libpng-dev

      - name: Install Python Dependencies
        run: |
          pip install -r requirements.txt

      - name: Run Monitor Script
        env:
          SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
          SUPABASE_KEY: ${{ secrets.SUPABASE_KEY }}
        run: python monitor.py

      - name: Commit and Push Changes
        run: |
          git config --global user.name "SniperBot"
          git config --global user.email "bot@noreply.github.com"
          git add assets/*.png README.md LATEST_ACTION.md
          git commit -m "📈 Auto-Update: Daily Dashboard" || echo "No changes to commit"
          git push
"""
    write_file(workflow_path, workflow_content)
    
    print("✅ WORKFLOW & .gitignore FILES UPDATED.")
    print("\n👉 NEXT STEPS:")
    print("1. Make sure your models are in the 'models/' folder and named correctly.")
    print("2. Commit and push ALL changes.")
    print("   Run these commands in your terminal:")
    print("   git add .")
    print("   git commit -m \"Fix: Update gitignore and workflow permissions\"")
    print("   git push")

if __name__ == "__main__":
    main()