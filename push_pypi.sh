#!/bin/bash

# Extract version number from __init__.py

# Extract version number
VERSION=$(awk -F"'" '/^__version__/ {print $2}' yaiv/__init__.py)
echo $VERSION

# Ensure we're up to date
git fetch

# Switch to dev branch and reset changes
git switch dev
git add -A
git commit -m "Version $VERSION"
git push private dev

# Switch to pip branch
git switch pip

# Merge changes from dev into pip, excluding yaiv/dev
git checkout dev -- . ':!yaiv/dev'

# Reset to clean staging area
git reset HEAD -- .
git status

# Create a commit with the version number
git commit -m "Merge dev into pip (excluding yaiv/dev) — Version $VERSION"

# Create a fake merge commit for bookkeeping to avoid confusion later
git merge -s ours dev

# Verify the differences (should only be yaiv/dev)
echo "Differences between pip and dev branches:"
git diff --name-only dev

# Push Changes
git push private pip

echo "Merge and push to pip branch completed with version $VERSION."

#BUILD THE PACKAGE
python3 -m pip install --upgrade build
python3 -m build

##UPLOADE IT TO PYPI
python3 -m pip install --upgrade twine
python3 -m twine upload dist/*
