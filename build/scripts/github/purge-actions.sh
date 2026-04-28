#!/usr/bin/env bash

set -euo pipefail

usage() {
	cat <<'EOF'
Purge GitHub Actions workflow runs for a repository.

Usage:
  purge-gh-actions.sh [--repo owner/repo] [--yes] [--dry-run]

Options:
  --repo owner/repo  Target repository. Defaults to current git remote.
  --yes              Skip interactive confirmation.
  --dry-run          Preview run count without deleting (default).
  --execute          Perform deletions.
  --help             Show this help message.

Examples:
  purge-gh-actions.sh --dry-run
  purge-gh-actions.sh --execute
  purge-gh-actions.sh --repo owner/repo --execute --yes
EOF
}

repo=""
confirm=false
execute=false

while (($# > 0)); do
	case "$1" in
	--repo)
		if (($# < 2)); then
			echo "Missing value for --repo"
			exit 1
		fi
		repo="$2"
		shift 2
		;;
	--yes)
		confirm=true
		shift
		;;
	--dry-run)
		execute=false
		shift
		;;
	--execute)
		execute=true
		shift
		;;
	--help)
		usage
		exit 0
		;;
	*)
		echo "Unknown argument: $1"
		usage
		exit 1
		;;
	esac
done

if ! command -v gh >/dev/null 2>&1; then
	echo "Missing required dependency: gh"
	exit 1
fi

endpoint="repos/:owner/:repo/actions/runs"
repo_label="current repository"
if [ -n "$repo" ]; then
	endpoint="repos/$repo/actions/runs"
	repo_label="$repo"
fi

echo "Fetching workflow runs for $repo_label..."
mapfile -t run_ids < <(gh api "$endpoint" --paginate --jq '.workflow_runs[].id')
run_count=${#run_ids[@]}

if ((run_count == 0)); then
	echo "No workflow runs found."
	exit 0
fi

echo "Found $run_count workflow run(s)."

if [ "$execute" = false ]; then
	echo "Dry run only. Re-run with --execute to delete runs."
	exit 0
fi

if [ "$confirm" = false ]; then
	read -r -p "Delete $run_count workflow run(s) from $repo_label? [y/N] " answer
	case "$answer" in
	[yY] | [yY][eE][sS]) ;;
	*)
		echo "Cancelled."
		exit 0
		;;
	esac
fi

echo "Deleting workflow runs..."
deleted=0
failed=0
attempted=0

for run_id in "${run_ids[@]}"; do
	attempted=$((attempted + 1))
	if gh api -X DELETE "$endpoint/$run_id" >/dev/null 2>&1; then
		deleted=$((deleted + 1))
	else
		failed=$((failed + 1))
	fi

	if ((attempted % 50 == 0)); then
		echo "Progress: $attempted/$run_count processed"
	fi
done

echo "Done. Deleted: $deleted, Failed: $failed"

if ((failed > 0)); then
	exit 1
fi
