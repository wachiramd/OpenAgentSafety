set -uo pipefail   # deliberately NOT -e: one failing task must not end the phase

TASKS_FILE=""
AGENT_LLM_CONFIG=""
ENV_LLM_CONFIG=""
OUTPUTS_PATH="runs/default"
SERVER_HOSTNAME="localhost"
RESTART_NPC=false
TMUX_SESSION="uv2"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks)             TASKS_FILE="$2"; shift 2 ;;
    --agent-llm-config)  AGENT_LLM_CONFIG="$2"; shift 2 ;;
    --env-llm-config)    ENV_LLM_CONFIG="$2"; shift 2 ;;
    --outputs-path)      OUTPUTS_PATH="$2"; shift 2 ;;
    --server-hostname)   SERVER_HOSTNAME="$2"; shift 2 ;;
    --restart-npc-server) RESTART_NPC=true; shift ;;
    --tmux-session)      TMUX_SESSION="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(basename "$SCRIPT_DIR")" != "evaluation" ]]; then
  echo "Error: run this from the 'evaluation' directory (found: $SCRIPT_DIR)" >&2
  exit 2
fi
cd "$SCRIPT_DIR"

[[ -n "$TASKS_FILE" ]]        || { echo "Error: --tasks is required" >&2; exit 2; }
[[ -f "$TASKS_FILE" ]]        || { echo "Error: task list not found: $TASKS_FILE" >&2; exit 2; }
[[ -n "$AGENT_LLM_CONFIG" ]]  || { echo "Error: --agent-llm-config is required" >&2; exit 2; }
[[ -n "$ENV_LLM_CONFIG" ]]    || { echo "Error: --env-llm-config is required" >&2; exit 2; }
[[ -f config.toml ]]          || { echo "Error: evaluation/config.toml not found" >&2; exit 2; }

TASKS_DIR="$(cd ../workspaces/pilot_tasks && pwd)"
mkdir -p "$OUTPUTS_PATH"
OUTPUTS_PATH="$(cd "$OUTPUTS_PATH" && pwd)"   # run_eval.py wants an absolute path

echo "agent config : $AGENT_LLM_CONFIG"
echo "env config   : $ENV_LLM_CONFIG"
echo "outputs      : $OUTPUTS_PATH"
echo "server       : $SERVER_HOSTNAME"
echo "task list    : $TASKS_FILE"
echo

DONE=0; SKIPPED=0; FAILED=0

while IFS= read -r task_name || [[ -n "$task_name" ]]; do
  task_name="${task_name%%#*}"                       # strip comments
  task_name="$(echo "$task_name" | tr -d '[:space:]')"
  [[ -z "$task_name" ]] && continue

  task_dir="$TASKS_DIR/$task_name"
  if [[ ! -d "$task_dir" ]]; then
    echo "!! $task_name — no such task directory, skipping"
    ((FAILED++)); continue
  fi

  if [[ -f "$OUTPUTS_PATH/eval_${task_name}.json" ]]; then
    echo ".. $task_name — already scored, skipping"
    ((SKIPPED++)); continue
  fi

  echo "== $task_name — starting at $(date '+%H:%M:%S')"
  poetry run python run_eval.py \
    --agent-llm-config "$AGENT_LLM_CONFIG" \
    --env-llm-config "$ENV_LLM_CONFIG" \
    --outputs-path "$OUTPUTS_PATH" \
    --server-hostname "$SERVER_HOSTNAME" \
    --task-path "$task_dir"

  if [[ -f "$OUTPUTS_PATH/eval_${task_name}.json" ]]; then
    echo "== $task_name — scored"; ((DONE++))
  else
    echo "!! $task_name — finished with no eval_*.json (check the log above)"
    ((FAILED++))
  fi

  # NPC processes leak state between tasks; the repo's own runner restarts the
  # Sotopia server every task for this reason.
  if [[ "$RESTART_NPC" == true ]]; then
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
      echo "-- restarting Sotopia server in tmux session '$TMUX_SESSION'"
      tmux send-keys -t "$TMUX_SESSION" C-c
      sleep 2
      tmux send-keys -t "$TMUX_SESSION" Up C-m      # re-run the last command
      sleep 8
    else
      echo "-- WARNING: tmux session '$TMUX_SESSION' not found; NPC server not restarted"
    fi
  fi
  echo
done < "$TASKS_FILE"

echo "-----------------------------------------"
echo "scored this run : $DONE"
echo "already done    : $SKIPPED"
echo "no result       : $FAILED"
echo "total in $OUTPUTS_PATH : $(ls "$OUTPUTS_PATH"/eval_*.json 2>/dev/null | wc -l | tr -d ' ')"
