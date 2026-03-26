#!/usr/bin/env bash
# run-agent.sh — launch an AI agent in an isolated bubblewrap sandbox
#
# Usage:
#   ./run-agent.sh --agent-id <id> --aws-profile <profile> --workspace <path> -- <command> [args...]
#
# Example:
#   ./run-agent.sh --agent-id agent-1 --aws-profile prod-account -- python agent.py
#   ./run-agent.sh --agent-id agent-2 --aws-credentials-file /secrets/agent2.creds -- bash
#
# AWS credentials are resolved in this order (first wins):
#   1. --aws-credentials-file  (explicit file)
#   2. --aws-profile           (named profile from host ~/.aws)
#   3. Env vars AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY already set

set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
AGENT_ID=""
AWS_PROFILE=""
AWS_CREDENTIALS_FILE=""
WORKSPACE=""
DOTFILES_DIR="${AGENT_SANDBOX_DOTFILES:-}"   # optional: dir of dotfiles to overlay
RUNTIME_BASE="${AGENT_SANDBOX_RUNTIME:-/run/agent-sandbox}"
SHARE_NET=true
COMMAND=()

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent-id)             AGENT_ID="$2";               shift 2 ;;
    --aws-profile)          AWS_PROFILE="$2";             shift 2 ;;
    --aws-credentials-file) AWS_CREDENTIALS_FILE="$2";   shift 2 ;;
    --workspace)            WORKSPACE="$2";               shift 2 ;;
    --dotfiles-dir)         DOTFILES_DIR="$2";            shift 2 ;;
    --no-network)           SHARE_NET=false;              shift   ;;
    --)                     shift; COMMAND=("$@");        break   ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# ── validate ──────────────────────────────────────────────────────────────────
[[ -z "$AGENT_ID" ]]    && { echo "ERROR: --agent-id is required" >&2; exit 1; }
[[ ${#COMMAND[@]} -eq 0 ]] && { echo "ERROR: command is required (after --)" >&2; exit 1; }

if [[ -z "$WORKSPACE" ]]; then
  WORKSPACE="$(pwd)"
  echo "INFO: --workspace not set, using current directory: $WORKSPACE"
fi
[[ -d "$WORKSPACE" ]] || { echo "ERROR: workspace does not exist: $WORKSPACE" >&2; exit 1; }

# ── per-agent runtime dir ─────────────────────────────────────────────────────
AGENT_RUNTIME="$RUNTIME_BASE/$AGENT_ID"
AGENT_AWS_DIR="$AGENT_RUNTIME/aws"
AGENT_TMP_DIR="$AGENT_RUNTIME/tmp"

cleanup() {
  rm -rf "$AGENT_RUNTIME"
}
trap cleanup EXIT

mkdir -p "$AGENT_AWS_DIR" "$AGENT_TMP_DIR"
chmod 700 "$AGENT_AWS_DIR"

# ── populate isolated ~/.aws ──────────────────────────────────────────────────
setup_aws_credentials() {
  if [[ -n "$AWS_CREDENTIALS_FILE" ]]; then
    # explicit file provided
    [[ -f "$AWS_CREDENTIALS_FILE" ]] || { echo "ERROR: credentials file not found: $AWS_CREDENTIALS_FILE" >&2; exit 1; }
    cp "$AWS_CREDENTIALS_FILE" "$AGENT_AWS_DIR/credentials"
    chmod 600 "$AGENT_AWS_DIR/credentials"
    echo "INFO: using explicit credentials file"

  elif [[ -n "$AWS_PROFILE" ]]; then
    # extract the named profile from the host credentials/config
    local host_creds="$HOME/.aws/credentials"
    local host_cfg="$HOME/.aws/config"

    if [[ -f "$host_creds" ]]; then
      extract_profile "$host_creds" "$AWS_PROFILE" "$AGENT_AWS_DIR/credentials"
    fi
    if [[ -f "$host_cfg" ]]; then
      extract_profile "$host_cfg" "profile $AWS_PROFILE" "$AGENT_AWS_DIR/config"
      # also try without the "profile " prefix (default profile)
      [[ -s "$AGENT_AWS_DIR/config" ]] || \
        extract_profile "$host_cfg" "$AWS_PROFILE" "$AGENT_AWS_DIR/config"
    fi
    [[ -s "$AGENT_AWS_DIR/credentials" ]] || \
      echo "WARN: no credentials found for profile '$AWS_PROFILE'" >&2
    echo "INFO: using aws profile '$AWS_PROFILE'"

  elif [[ -n "${AWS_ACCESS_KEY_ID:-}" ]]; then
    # write env-var credentials into isolated file so refreshes stay isolated
    cat > "$AGENT_AWS_DIR/credentials" <<EOF
[default]
aws_access_key_id = ${AWS_ACCESS_KEY_ID}
aws_secret_access_key = ${AWS_SECRET_ACCESS_KEY}
${AWS_SESSION_TOKEN:+aws_session_token = ${AWS_SESSION_TOKEN}}
EOF
    chmod 600 "$AGENT_AWS_DIR/credentials"
    echo "INFO: using credentials from environment variables"

  else
    echo "WARN: no AWS credentials configured for agent '$AGENT_ID'" >&2
  fi
}

# Extract a [section] block from an ini-style file into an output file.
# Usage: extract_profile <src> <section-name> <dest>
extract_profile() {
  local src="$1" section="$2" dest="$3"
  awk -v sec="[$section]" '
    /^\[/ { found = ($0 == sec) }
    found  { print }
    found && /^\[/ && NR > 1 && $0 != sec { exit }
  ' "$src" > "$dest"
  chmod 600 "$dest"
}

setup_aws_credentials

# ── build bwrap args ──────────────────────────────────────────────────────────
BWRAP_ARGS=(
  # isolated /tmp so agents can't see each other's temp files
  --tmpfs /tmp

  # isolated ~/.aws bound from per-agent runtime dir (read-write so refresh works)
  --bind "$AGENT_AWS_DIR" "$HOME/.aws"

  # workspace is the only writable project directory
  --bind "$WORKSPACE" /workspace

  # read-only system paths
  --ro-bind /usr /usr
  --ro-bind /lib /lib
  --ro-bind-try /lib64 /lib64
  --ro-bind /bin /bin
  --ro-bind /sbin /sbin
  --ro-bind /etc/resolv.conf /etc/resolv.conf
  --ro-bind /etc/ssl /etc/ssl
  --ro-bind /etc/passwd /etc/passwd
  --ro-bind /etc/group /etc/group

  # proc/dev
  --proc /proc
  --dev /dev

  # isolate all namespaces except network (optional)
  --unshare-pid
  --unshare-ipc
  --unshare-uts

  # set working directory inside sandbox
  --chdir /workspace
)

# optional dotfiles overlay (read-only)
if [[ -n "$DOTFILES_DIR" ]]; then
  [[ -d "$DOTFILES_DIR" ]] || { echo "ERROR: dotfiles dir not found: $DOTFILES_DIR" >&2; exit 1; }
  for f in "$DOTFILES_DIR"/.*; do
    [[ -e "$f" ]] || continue
    name="$(basename "$f")"
    [[ "$name" == "." || "$name" == ".." || "$name" == ".aws" ]] && continue
    BWRAP_ARGS+=(--ro-bind "$f" "$HOME/$name")
  done
fi

# network
if $SHARE_NET; then
  BWRAP_ARGS+=(--share-net)
else
  BWRAP_ARGS+=(--unshare-net)
fi

# ── launch ────────────────────────────────────────────────────────────────────
echo "INFO: launching agent '$AGENT_ID' in sandbox (workspace: $WORKSPACE)"
exec bwrap "${BWRAP_ARGS[@]}" -- "${COMMAND[@]}"
