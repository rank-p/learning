# agent-sandbox

Bubblewrap-based sandbox for running AI agents in parallel with isolated
filesystems and per-agent AWS credentials.

## Problem solved

When multiple agents run simultaneously and refresh AWS credentials, they all
write to `~/.aws/credentials` — overwriting each other and causing requests
to land in the wrong account. This sandbox gives each agent its own isolated
`~/.aws` directory, so credential refreshes never collide.

## Requirements

- [`bubblewrap`](https://github.com/containers/bubblewrap) (`bwrap`)
  - Debian/Ubuntu: `apt install bubblewrap`
  - Fedora/RHEL: `dnf install bubblewrap`
  - Arch: `pacman -S bubblewrap`

## Single agent

```bash
chmod +x run-agent.sh

# Using a named AWS profile from ~/.aws
./run-agent.sh \
  --agent-id    my-agent \
  --aws-profile prod-account \
  --workspace   /path/to/project \
  -- python agent.py

# Using an explicit credentials file
./run-agent.sh \
  --agent-id             my-agent \
  --aws-credentials-file /secrets/prod.credentials \
  --workspace            /path/to/project \
  -- python agent.py

# Using current AWS env vars (AWS_ACCESS_KEY_ID etc.)
./run-agent.sh \
  --agent-id  my-agent \
  --workspace /path/to/project \
  -- python agent.py

# With dotfiles overlaid (read-only) and no network
./run-agent.sh \
  --agent-id    my-agent \
  --aws-profile prod-account \
  --workspace   /path/to/project \
  --dotfiles-dir ~/.dotfiles \
  --no-network \
  -- python agent.py
```

## Multiple agents in parallel

1. Copy and edit the manifest:

```bash
cp agents.conf.example agents.conf
# edit agents.conf with your agent IDs, profiles, workspaces, and commands
```

2. Run:

```bash
chmod +x run-agents-parallel.sh
./run-agents-parallel.sh agents.conf
```

Logs are written to `/tmp/agent-sandbox-logs/<agent-id>.log` by default.
Override with `AGENT_SANDBOX_LOGS=/your/path`.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `AGENT_SANDBOX_RUNTIME` | `/run/agent-sandbox` | Base dir for per-agent runtime state (aws dirs, tmp) |
| `AGENT_SANDBOX_LOGS` | `/tmp/agent-sandbox-logs` | Log output dir (parallel runner) |
| `AGENT_SANDBOX_DOTFILES` | _(unset)_ | Default dotfiles dir to overlay into all sandboxes |

## How isolation works

```
Host                          Sandbox (per agent)
────                          ───────────────────
/run/agent-sandbox/
  agent-1/aws/   ──bind──►   ~/.aws/          (read-write, isolated)
  agent-2/aws/   ──bind──►   ~/.aws/          (read-write, isolated)

/workspaces/project-a  ──►   /workspace       (read-write)
/usr, /lib, /bin, ...  ──►   same paths       (read-only)
                             /tmp             (tmpfs, isolated)
```

Credential refreshes inside agent-1's sandbox write to
`/run/agent-sandbox/agent-1/aws/credentials` — completely separate from
agent-2's credentials. The host's `~/.aws` is never touched.
