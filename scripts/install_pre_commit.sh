#!/bin/sh
# Install the root pre-commit guard. Does not call the archived CDAP audit.
set -eu
root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
hook="$root/.git/hooks/pre-commit"
mkdir -p "$root/.git/hooks"
cat > "$hook" << EOF
#!/bin/sh
cd "$root" || exit 1
exec python3 scripts/pre_commit_guard.py
EOF
chmod +x "$hook" scripts/pre_commit_guard.py
echo "installed $hook"
