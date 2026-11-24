#!/bin/sh
# Install the root pre-commit guard. Does not call the archived CDAP audit.
# The hook runs the first interpreter that actually starts: the project venv,
# then python3, then python (on Windows, python3 can be the Microsoft Store
# stub, which exits non-zero and would block every commit).
set -eu
root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
hook="$root/.git/hooks/pre-commit"
mkdir -p "$root/.git/hooks"
cat > "$hook" << EOF
#!/bin/sh
cd "$root" || exit 1
for py in .venv/Scripts/python.exe .venv/bin/python python3 python; do
    if "\$py" -c "" >/dev/null 2>&1; then
        exec "\$py" scripts/pre_commit_guard.py
    fi
done
echo "pre-commit: no working Python interpreter found" >&2
exit 1
EOF
chmod +x "$hook" scripts/pre_commit_guard.py
echo "installed $hook"
