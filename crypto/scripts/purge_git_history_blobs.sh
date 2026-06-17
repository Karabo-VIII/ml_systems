#!/bin/bash
# Path B: destructive history rewrite to remove the 119 GB of chimera/model blobs.
#
# DO NOT RUN THIS BLINDLY. It rewrites all 637 commits, changes every commit
# hash, and requires a force-push to origin. All clones become invalid;
# anyone collaborating must re-clone after.
#
# Why this exists: at some point pre-2026-04 commits added
# data/processed/chimera*/*.parquet (each 1-2 GB). Even though the current
# .gitignore catches these, the blobs persist in .git/objects/. To reclaim
# the 119 GB we must purge them from history.
#
# Pre-flight checklist:
#   [ ] Verify only YOU are using the repo (no other clones or CI consumers).
#       $ git branch -a   (should show only master + remotes/origin/master)
#   [ ] Have a fresh on-disk backup of the repo (just in case).
#   [ ] Ensure git-filter-repo is installed: `pip install git-filter-repo`
#   [ ] V1.0 training is NOT mid-flight (process tree should be clean).
#
# Sequence:
#   1. Install git-filter-repo
#   2. Remove data/processed/, models/, and binary-extension blobs from ALL history
#   3. Repack and prune
#   4. Force-push to origin (DESTRUCTIVE)
#   5. Verify .git size dropped from 119 GB -> ~50-200 MB
#
# After this completes: re-clone the repo on any other machine. Old hashes
# are invalid.

set -euo pipefail

echo "============================================================"
echo "  GIT HISTORY BLOB PURGE -- DESTRUCTIVE"
echo "============================================================"
echo ""
echo "Current .git size:"
du -sh .git 2>/dev/null
echo ""
echo "This script will REMOVE the following from ALL git history:"
echo "  - data/processed/   (all sub-paths)"
echo "  - models/           (all sub-paths)"
echo "  - *.parquet *.pt *.pth *.pkl *.pickle *.bin *.npy *.npz *.h5 *.hdf5"
echo "  - *.feather *.arrow"
echo ""
echo "All 637 commit hashes will change. Force-push required."
echo ""
read -r -p "Type 'PURGE' to continue, anything else to abort: " confirm
if [ "$confirm" != "PURGE" ]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "[1/5] Installing git-filter-repo (if not present)..."
if ! python -c "import git_filter_repo" 2>/dev/null; then
    pip install git-filter-repo
fi

echo ""
echo "[2/5] Purging path-based blobs (data/processed/, models/)..."
git filter-repo --path data/processed/ --invert-paths --force
git filter-repo --path models/ --invert-paths --force

echo ""
echo "[3/5] Purging extension-based blobs..."
git filter-repo \
    --path-glob '*.parquet' \
    --path-glob '*.pt' \
    --path-glob '*.pth' \
    --path-glob '*.pkl' \
    --path-glob '*.pickle' \
    --path-glob '*.bin' \
    --path-glob '*.npy' \
    --path-glob '*.npz' \
    --path-glob '*.h5' \
    --path-glob '*.hdf5' \
    --path-glob '*.feather' \
    --path-glob '*.arrow' \
    --invert-paths --force

echo ""
echo "[4/5] Repacking and pruning..."
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo ""
echo "Post-purge .git size:"
du -sh .git 2>/dev/null
echo ""

echo "[5/5] Force-push to origin (DESTRUCTIVE) ..."
echo ""
echo "Filter-repo removes the 'origin' remote on completion (safety feature)."
echo "Re-add it and force-push:"
echo ""
echo "  git remote add origin https://github.com/Karabo-VIII/v4_crypto_stystem.git"
echo "  git push --force --all origin"
echo "  git push --force --tags origin"
echo ""
echo "After successful force-push, any other clone of this repo MUST re-clone."
echo "Old commit hashes are invalid."
echo ""
echo "Done. Verify the GitHub repo size dropped. (Push not auto-executed; do it"
echo "manually after sanity-checking the local state.)"
