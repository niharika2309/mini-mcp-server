#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run.sh - start the whole system with one command.
#
# The MCP server (server.py) is not run separately: the agent spawns it as a
# child process and speaks MCP over stdio. This script boots the venv, runs
# a server+discovery sanity check, then launches the agent in the mode you
# pick.
#
# Usage:
#   ./run.sh                 # browser UI at http://127.0.0.1:8765
#   ./run.sh web [PORT]      # browser UI on a custom port
#   ./run.sh cli             # terminal REPL
#   ./run.sh "any question"  # one-shot: answer and exit
#   ./run.sh demo            # offline REPL with the scripted mock LLM (no keys)
#   ./run.sh demo-web        # browser UI on the mock LLM
#   ./run.sh list            # start the server, print discovered tools, exit
#   ./run.sh test            # run the pytest suite
#   ./run.sh stop            # stop a running browser UI
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=".venv/bin/python"
MODE="${1:-web}"
PORT="${2:-${PORT:-8765}}"

# --- 1) Environment ---------------------------------------------------------
if [ ! -x "$PYTHON" ]; then
  echo "==> No .venv found - creating it and installing requirements"
  if command -v uv >/dev/null 2>&1; then
    uv venv .venv
    uv pip install -p "$PYTHON" -r requirements.txt
  else
    python3 -m venv .venv
    "$PYTHON" -m pip install -q -r requirements.txt
  fi
fi

if [ ! -f .env ]; then
  echo "!! .env is missing. Copy .env.example to .env and fill in:"
  echo "   OPENAI_BASE_URL / OPENAI_API_KEY / MODEL  (or LLM_*) and GITHUB_TOKEN"
  echo "   (You can still try './run.sh demo' key-free.)"
  [ "$MODE" = "demo" ] || [ "$MODE" = "demo-web" ] || exit 1
fi

ui_running() { curl -s -m 2 "http://127.0.0.1:${PORT}/api/info" >/dev/null 2>&1; }

# --- 2) Modes that don't need the server ------------------------------------
case "$MODE" in
  stop)
    pkill -f "python -m agent --web" && echo "==> Browser UI stopped" || echo "==> Nothing was running"
    exit 0 ;;
  test)
    exec "$PYTHON" -m pytest ;;
esac

# --- 3) Sanity check: boot the MCP server + discover tools ------------------
echo "==> Starting MCP server child and verifying tool discovery"
if ! "$PYTHON" -m agent --list-tools >/tmp/agent-list-tools.out 2>&1; then
  echo "!! Server failed to start or register tools:"
  tail -5 /tmp/agent-list-tools.out
  exit 1
fi
TOOLS=$(grep -oE "[0-9]+ tools discovered" /tmp/agent-list-tools.out | grep -oE "^[0-9]+" || true)
echo "==> MCP server OK - ${TOOLS:-?} tools discovered"

# --- 4) Run the agent in the requested mode ---------------------------------
case "$MODE" in
  web)
    if ui_running; then
      echo "==> UI already running at http://127.0.0.1:${PORT} (use './run.sh stop' first to restart)"
      exit 0
    fi
    echo "==> Browser UI: http://127.0.0.1:${PORT}  (Ctrl+C to stop)"
    exec "$PYTHON" -m agent --web --port "$PORT" ;;
  demo-web)
    if ui_running; then
      echo "==> UI already running at http://127.0.0.1:${PORT} (use './run.sh stop' first to restart)"
      exit 0
    fi
    echo "==> Browser UI (offline mock LLM): http://127.0.0.1:${PORT}  (Ctrl+C to stop)"
    exec "$PYTHON" -m agent --web --demo --port "$PORT" ;;
  cli)
    exec "$PYTHON" -m agent ;;
  demo)
    exec "$PYTHON" -m agent --demo ;;
  list)
    cat /tmp/agent-list-tools.out ;;
  *)
    # Anything else is treated as a one-shot question.
    exec "$PYTHON" -m agent "$@" ;;
esac
