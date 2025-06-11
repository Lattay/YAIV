#!/bin/bash

#=======================================================================
#                               push_pypi.sh
#=======================================================================
#
# Description:
#   This script automates the process of merging branches, updating
#   version numbers, building a Python package, and uploading to PyPI.
#
# Usage:
#   $ ./push_pypi.sh [options]
#
# Options:
#   -h, --help       Display this help message.
#
# Example:
#   ./push_pypi.sh
#
#=======================================================================
# Author: Martin Gutierrez-Amigo
# Created: 2025-06-10
#=======================================================================

# Function: print_help
print_help() {
    echo "push_pypi.sh - Automates merging branches and uploading packages to PyPI."
    echo "Usage: $ ./push_pypi.sh [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Display this help message."
    echo ""
    echo "Example:"
    echo "  ./push_pypi.sh"
}

# Parse command-line arguments
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    print_help
    exit 0
fi

# Main script logic

# Extract version number from __init__.py
VERSION=$(awk -F"'" '/^__version__/ {print $2}' yaiv/__init__.py)
echo "Extracted Version: $VERSION"

# Confirm version
read -p "Is the version number correct? (yes/no): " confirm_version
if [[ "$confirm_version" != "yes" ]]; then
    echo "Version not confirmed, exiting."
    exit 1
fi

# Ensure we're up to date
echo Fetching...
git fetch

# Switch to dev branch and push changes
read -p "Do you want to proceed with updating the dev branch? (yes/no): " proceed
if [[ "$proceed" == "yes" ]]; then
    git switch dev
    git add -A
    git commit -m "NEW UPDATE v$VERSION"
    git push private dev
else
    echo "Skipping dev branch update."
fi
echo "======================================================================="
echo

# Merge dev into pip
read -p "Do you want to proceed with merging dev into pip? (yes/no): " proceed
if [[ "$proceed" == "yes" ]]; then
    # Switch to pip branch
    git switch pip
    # Merge changes from dev into pip, excluding yaiv/dev
    git checkout dev -- . ':!yaiv/dev'
    git status

    # Verify the differences (should only be yaiv/dev)
    echo "Differences between pip and dev branches:"
    git diff --name-only dev

    # Confirm differences before pushing
    read -p "Are the differences as expected (only dev/ files) and you want to merge? (yes/no): " confirm_diff
    if [[ "$confirm_diff" != "yes" ]]; then
        echo "Differences not as expected, exiting."
        exit 1
    fi

    # Create a commit with the version number
    git commit -m "Merge dev into pip (excluding yaiv/dev) — Version $VERSION"
    # Create a fake merge commit for bookkeeping
    git merge -s ours dev -m "Merge dev into pip (excluding yaiv/dev) — Version $VERSION"
    # Push Changes to pip branch
    git tag -a "v$VERSION" -m "Release version $VERSION"
    git push private pip
    echo "Merge and push to pip branch completed with version $VERSION."
else
    echo "Skipping merge from dev into pip."
fi
echo "======================================================================="
echo

# Merge pip into main (public)
read -p "Do you want to proceed with updating the main branch from pip? (yes/no): " proceed
if [[ "$proceed" == "yes" ]]; then
    # Update main branch from pip and push
    git switch main
    git checkout pip -- . ':!new_test_env.sh' ':!push_pypi.sh' ':!pyproject.toml'
    git commit -m "Merge pip into main — Version $VERSION"
    git merge -s ours pip -m "Merge pip into main — Version $VERSION"
    git push private main
    git push public main
    echo "Merge and push to main branch completed with version $VERSION."
else
    echo "Skipping update of main branch from pip."
fi
echo "======================================================================="
echo

# Build  PyPi package
read -p "Do you want to build the package? (yes/no): " proceed
if [[ "$proceed" == "yes" ]]; then
    # Build the package
    git switch pip
    python3 -m pip install --upgrade build
    python3 -m build
    echo "Package build completed."
else
    echo "Skipping package build."
fi
echo "======================================================================="
echo

# Publisy PyPi package
read -p "Do you want to upload the package to PyPI? (yes/no): " proceed
if [[ "$proceed" == "yes" ]]; then
    # Upload it to PyPI
    python3 -m pip install --upgrade twine
    python3 -m twine upload dist/*
    echo "Package uploaded."
    git switch dev
else
    echo "Skipping upload to PyPI."
fi
echo "======================================================================="
echo
echo "DONE 🎉"
