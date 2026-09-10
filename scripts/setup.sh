#!/usr/bin/env bash
# NoteFactory public setup (POSIX shell)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON=""
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON="$candidate"
    break
  fi
done
if [[ -z "$PYTHON" ]]; then
  echo "Python 3.10+ was not found." >&2
  exit 1
fi

[[ -d .venv ]] || "$PYTHON" -m venv .venv
VENV_PY=".venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "Could not find Python inside .venv." >&2
  exit 1
fi

"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install -r requirements.txt

if [[ ! -f .env.local ]]; then
  cp .env.example .env.local
  echo "Created .env.local from .env.example. Add GEMINI_API_KEY before generating a note."
else
  echo "Kept existing .env.local."
fi

echo "Setup complete. Run: .venv/bin/python note_pipe.py <sipher.json> --out notes_out"
