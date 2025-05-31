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

# Edit this and continue reading
PROJECT_ROOT="/abs/path/to/gerpa/repo"

PACKAGE_NAME="gerpa"
MODULE_NAME="main"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Now run the module
python3 -m "${PACKAGE_NAME}.${MODULE_NAME}" "$@"

# To make this executable (macOS / Linux / WSL2):
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
# 5. Make sure ~/.local/bin is in your PATH if it's not already there
#    (e.g., ~/.bashrc, ~/.zshrc, or ~/.profile - depending on your shell and OS):
#    for f in ~/.bashrc ~/.zshrc ~/.profile; do [ -f "$f" ] && grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' "$f" || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$f"; done
# 6. Reload your shell config:
#    source ~/.bashrc  # or 'source ~/.zshrc' or restart your terminal
```
