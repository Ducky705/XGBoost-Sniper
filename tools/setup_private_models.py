import os
import subprocess
import shutil

def run_cmd(cmd):
    print(f"Executing: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.stdout

def setup():
    print("🚀 XGBoost-Sniper Model Migration Tool")
    
    repo_url = input("Enter your private GitHub model repository URL (e.g., git@github.com:Ducky705/XGBoost-Sniper-Models.git): ").strip()
    if not repo_url:
        print("❌ URL is required.")
        return

    # 1. Backup
    if os.path.exists("models") and not os.path.islink("models"):
        print("📦 Backing up current models to 'models_backup'...")
        if os.path.exists("models_backup"):
            shutil.rmtree("models_backup")
        shutil.copytree("models", "models_backup")
    
    # 2. Create temp dir for private repo
    temp_dir = "../XGBoost-Sniper-Models-Temp"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    print("🚚 Copying models to private repo structure...")
    for item in os.listdir("models"):
        shutil.copy2(os.path.join("models", item), os.path.join(temp_dir, item))
    
    # 3. Initialize and Push Private Repo
    print("🛰️ Initializing private model repository...")
    os.chdir(temp_dir)
    run_cmd("git init")
    run_cmd(f"git remote add origin {repo_url}")
    run_cmd("git add .")
    run_cmd('git commit -m "feat: Initial model commit"')
    run_cmd("git branch -M main")
    print(f"⚠️  Please ensure you have created '{repo_url}' on GitHub.")
    push = input("Ready to push models to private repo? (y/n): ")
    if push.lower() == 'y':
        run_cmd("git push -u origin main")
    
    # 4. Add Submodule to Public Repo
    os.chdir("../XGBoost-Sniper")
    print("🔗 Linking private repo as submodule...")
    
    # Remove existing models folder to make room for submodule
    if os.path.exists("models") and not os.path.islink("models"):
        shutil.rmtree("models")
    
    run_cmd(f"git submodule add {repo_url} models")
    run_cmd("git add .gitmodules models")
    run_cmd('git commit -m "feat: Integrated private models via git submodule"')
    
    print("\n✨ Migration Complete!")
    print("1. Your public repo now has a 'models' submodule pointing to your private repo.")
    print("2. Your models are safe in a separate private repository.")
    print("3. Check 'models_backup' if anything went wrong.")

if __name__ == "__main__":
    setup()
