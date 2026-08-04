"""
Auto-commits and pushes output files to GitHub, gated on GH_TOKEN and
GH_REPO environment variables. Silently no-ops if either is missing.
"""

import os
import subprocess
from typing import List


def _run(cmd, cwd):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    safe_cmd = [c if c != os.environ.get("GH_TOKEN", object()) else "***" for c in cmd]
    print(f"$ {' '.join(safe_cmd)}")
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    return result


def push_outputs_to_github(repo_root: str, file_paths: List[str], commit_message: str) -> None:
    gh_token = os.environ.get("GH_TOKEN")
    github_repo = os.environ.get("GH_REPO")

    if not gh_token or not github_repo:
        print("[git_push] GH_TOKEN / GH_REPO not set — skipping auto-push.")
        return

    print("[git_push] Auto-push enabled — committing and pushing output files...")
    try:
        _run(["git", "config", "user.email", "kaggle-bot@example.com"], cwd=repo_root)
        _run(["git", "config", "user.name", "Kaggle Auto Push"], cwd=repo_root)

        remote_url = f"https://{gh_token}@github.com/{github_repo}.git"
        _run(["git", "remote", "set-url", "origin", remote_url], cwd=repo_root)

        rel_paths = [os.path.relpath(p, repo_root) for p in file_paths]
        status = subprocess.run(["git", "status", "--porcelain"] + rel_paths, cwd=repo_root, capture_output=True, text=True)
        if not status.stdout.strip():
            print("[git_push] No changes to commit.")
            return

        _run(["git", "add"] + rel_paths, cwd=repo_root)
        _run(["git", "commit", "-m", commit_message], cwd=repo_root)
        _run(["git", "push"], cwd=repo_root)
        print("[git_push] Push complete.")
    except Exception as e:
        print(f"[git_push] WARNING: auto-push failed, continuing without it: {e}")
