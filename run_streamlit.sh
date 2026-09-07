#!/usr/bin/env bash
set -e

export PATH="$HOME/.local/bin:$PATH"
exec python3 -m streamlit run main.py "$@"
