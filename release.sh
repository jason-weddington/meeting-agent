#!/usr/bin/env bash
# Cut a release: bump version, update CHANGELOG + uv.lock, tag, push to
# origin + github. Run at meaningful boundaries — not every commit.
#
# meeting-agent is a local CLI, not a deployed service, so there is no
# deploy step. If that changes, append the deploy invocation here.

set -euo pipefail

if [[ "$(git rev-parse --abbrev-ref HEAD)" != "main" ]]; then
    echo "error: release must run on main (you're on $(git rev-parse --abbrev-ref HEAD))" >&2
    exit 1
fi

if ! git diff-index --quiet HEAD --; then
    echo "error: working tree not clean — commit or stash before releasing" >&2
    exit 1
fi

git fetch origin --tags

echo "Bumping version + tagging via semantic-release..."
uv run semantic-release version --no-push --no-vcs-release

echo "Pushing main + tags to origin..."
git push origin main --tags

echo "Pushing main + tags to github..."
git push github main --tags

echo "Release done. Tag: $(git describe --tags --abbrev=0)"
