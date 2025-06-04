```bash
#!/bin/sh

# Notice: This package is experimental and not recommended
# for system-wide installation. Therefore, please follow
# the below instructions to make use of it once you have
# cloned the repository from GitHub.

# This file serves both as the README file and as a script
# template which makes GERPA usable as a portable CLI app.

# First, install dependencies and set up environment / packaging.
# Usage: You may use Makefile for the poetry / conda set-up
# (this is a versatile and recommended way). Alternatively, you may
# opt to simply go with the usual: pip install -r requirements.txt

# Edit the following line and continue reading
PROJECT_ROOT="/abs/path/to/gerpa/repo"

# Run these commands to locate Python
PACKAGE_NAME="gerpa" && MODULE_NAME="main"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
# Uncomment the following line if you went with a conda set-up
#eval "$(conda shell.bash hook)" && conda activate gerpa

# Now run the module
python3 -m "${PACKAGE_NAME}.${MODULE_NAME}" "$@"

# Install the 'gerpa' command (macOS / Linux / WSL2):
# (this will be available for the current user only;
#  no root or administrator access is needed):
# 0. Ensure the 'untracked' directory exists:
#    mkdir -p ./untracked
# 1. Copy the README.md file to ./untracked/gerpa,
#    removing the first line (```bash) and the last line (```):
#    sed '1d;$d' README.md > ./untracked/gerpa
# 2. Edit PROJECT_ROOT in ./untracked/gerpa manually.
# 3. Copy it to a directory in the local PATH:
#    mkdir -p "$HOME/.local/bin"
#    cp ./untracked/gerpa "$HOME/.local/bin/gerpa"
# 4. Make the script executable:
#    chmod +x "$HOME/.local/bin/gerpa"
# 5. Create an alias to make the command available globally:
#    echo 'alias gerpa="$HOME/.local/bin/gerpa"' >> ~/.bashrc
#    # For zsh users: echo 'alias gerpa="$HOME/.local/bin/gerpa"' >> ~/.zshrc
#    # For fish users: echo 'alias gerpa="$HOME/.local/bin/gerpa"' >> ~/.config/fish/config.fish
# 6. Reload your shell config:
#    source ~/.bashrc  # or 'source ~/.zshrc' for zsh
#    # or restart your terminal
#    # For fish: No manual reload needed (fish automatically sources config.fish)
###################################################
# 7. #### You may now run: gerpa ##################
###################################################
######### This should print usage instructions. ###
###################################################
# 8. To uninstall the command when necessary:
#    rm "$HOME/.local/bin/gerpa"
#    # Remove the alias from your shell config:
#    # For bash: sed -i '/alias gerpa=/d' ~/.bashrc
#    # For zsh: sed -i '/alias gerpa=/d' ~/.zshrc  
#    # For fish: sed -i '/alias gerpa=/d' ~/.config/fish/config.fish
#    # Then reload: source ~/.bashrc (or restart terminal)
```
