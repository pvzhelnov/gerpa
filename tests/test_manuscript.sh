#!/bin/bash
set -x
export PS4='+ $(date "+%Y-%m-%d %H:%M:%S")\t '

# --- Configuration ---
# echo 'PROJECT_ROOT="/app/gerpa"' > .env  # for Jules
source .env
LOG_DIR="$PROJECT_ROOT/tests/logs"
LOG_FILE="$LOG_DIR/test_manuscript.log"

# --- Logging Setup ---
mkdir -p "$LOG_DIR"
TMP_LOG=$(mktemp)
exec &> >(tee -a "$TMP_LOG")
echo "TRACE: set -x executed."
echo "TRACE: export PS4='+ \$(date \"+%Y-%m-%d %H:%M:%S\")\\t '"
echo "TRACE: source .env"
echo "TRACE: PROJECT_ROOT set to: $PROJECT_ROOT"
echo "TRACE: LOG_DIR set to: $LOG_DIR"
echo "TRACE: LOG_FILE set to: $LOG_FILE"
echo "TRACE: --- Logging setup commands (pre-logging) ---"
echo "TRACE: mkdir -p \"$LOG_DIR\""
echo "TRACE: TMP_LOG=$TMP_LOG"
echo "TRACE: exec &> >(tee -a \"$TMP_LOG\")"
echo "TRACE: --------------------------------------------"

# --- Timestamping & Execution Setup ---
# Start printing commands immediately.
# Exit immediately if a command exits with a non-zero status.
set -e

# --- Cleanup ---
cleanup() {
  echo "--- Cleaning up ---"
  rm -rf "$PROJECT_ROOT/untracked"
  rm -f "$HOME/.local/bin/gerpa"
  rm -rf "$PROJECT_ROOT/sample_project"
  echo "Cleanup complete."
  cat "$TMP_LOG" > "$LOG_FILE"
  rm "$TMP_LOG"
}
trap cleanup EXIT

# --- Main script ---
echo "--- Starting gerpa end-to-end test ---"

# 1. Check for Python 3
echo "--- Checking for Python 3 ---"
if ! command -v python3 &> /dev/null; then
  echo "Error: python3 is not installed or not in PATH."
  exit 1
fi
python3 --version
echo "Python 3 found."

# 2. Install dependencies
echo "--- Installing dependencies ---"
pip install -r "$PROJECT_ROOT/requirements.txt"
pip install quarto-cli
echo "Dependencies installed."

# 3. Install gerpa
echo "--- Installing gerpa ---"
BIN_DIR="$HOME/.local/bin"
mkdir -p "$BIN_DIR"
export PATH="$BIN_DIR:$PATH"
mkdir -p "$PROJECT_ROOT/untracked"
(cd "$PROJECT_ROOT" && sed '1d;$d' README.md > untracked/gerpa)
sed -i "s|PROJECT_ROOT=\"/abs/path/to/gerpa/repo\"|PROJECT_ROOT=\"$PROJECT_ROOT\"|g" "$PROJECT_ROOT/untracked/gerpa"
cp "$PROJECT_ROOT/untracked/gerpa" "$BIN_DIR/gerpa"
chmod +x "$BIN_DIR/gerpa"
echo "--- gerpa installed successfully ---"

# 4. Test gerpa command
echo "--- Testing gerpa command ---"
(cd "$PROJECT_ROOT" && gerpa init sample_project --manuscript)
if [ ! -d "$PROJECT_ROOT/sample_project" ]; then
  echo "Error: sample_project directory not created."
  exit 1
fi
echo "sample_project directory created."
if [ ! -f "$PROJECT_ROOT/sample_project/README.md" ]; then
  echo "Error: sample_project/README.md not found."
  exit 1
fi
echo "sample_project/README.md found."

if [ ! -d "$PROJECT_ROOT/sample_project/manuscript" ]; then
    echo "Error: sample_project/manuscript directory not created."
    exit 1
fi
echo "sample_project/manuscript directory created."

if [ ! -f "$PROJECT_ROOT/sample_project/manuscript/build/makerepref.py" ]; then
    echo "Error: sample_project/manuscript/build/makerepref.py not found."
    exit 1
fi
echo "sample_project/manuscript/build/makerepref.py found."

echo "--- Running makerepref.py script ---"
(cd "$PROJECT_ROOT/sample_project" && python manuscript/build/makerepref.py --bibmerge --verbose)

if [ ! -f "$PROJECT_ROOT/sample_project/manuscript.docx" ]; then
    echo "Error: sample_project/manuscript.docx not found."
    exit 1
fi
echo "sample_project/manuscript.docx found."


echo "--- gerpa command test passed ---"

echo "--- gerpa end-to-end test finished successfully ---"
