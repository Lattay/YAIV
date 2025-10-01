#!/bin/bash
#
#=======================================================================
#                               pytest_wrap.sh
#=======================================================================
#
# Description:
#   Convenience wrapper for pytest with matplotlib image comparison.
#   - Default: run pytest with --mpl, passing through any extra args.
#   - Generate: create/update baseline images into a directory (default tests/figs).
# # Usage: #   $ pytest_wrap.sh [options] [pytest-args...]
#
# Options:
#   -h, --help           Display this help message.
#   -g, --generate       Generate baseline images instead of comparing.
#   -b, --baseline DIR   Baseline images directory (default: tests/figs).
#
# Arguments:
#   [pytest-args...]     Any additional arguments passed to pytest (e.g., -k, -q).
#
# Examples:
#   # Run comparisons against baselines
#   pytest_wrap.sh
#   pytest_wrap.sh -k test/test1.py
#
#   # Generate (or update) baseline images into tests/figs
#   pytest_wrap.sh --generate
#   pytest_wrap.sh -g -b tests/figs -k tests/plot_test.py
#
#=======================================================================
# Author: Martin Gutierrez-Amigo
# Created: 2025-10-01
#=======================================================================

set -euo pipefail

print_help() {
    echo "pytest_wrap.sh - Wrapper for pytest with matplotlib image comparison."
    echo "Usage: $ pytest_wrap.sh [options] [pytest-args...]"
    echo ""
    echo "Options:"
    echo "  -h, --help           Display this help message."
    echo "  -g, --generate       Generate baseline images instead of comparing."
    echo "  -b, --baseline DIR   Baseline images directory (default: tests/figs)."
    echo ""
    echo "Arguments:"
    echo "  [pytest-args...]     Any additional arguments passed to pytest (e.g., -k, -q, test selection)."
    echo ""
    echo "Examples:"
    echo "  pytest_wrap.sh"
    echo "  pytest_wrap.sh -k test_bands_image_and_structure_cases"
    echo "  pytest_wrap.sh --generate"
    echo "  pytest_wrap.sh -g -b tests/figs -k test_bands_image_and_structure_cases"
}

# Main script logic
MODE="run"                 # run or generate
BASELINE_DIR="tests/figs"
PYTEST_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            print_help
            exit 0
            ;;
        -g|--generate)
            MODE="generate"
            shift
            ;;
        -b|--baseline)
            if [[ $# -lt 2 ]]; then
                echo "Error: --baseline requires a directory argument" >&2
                exit 1
            fi
            BASELINE_DIR="$2"
            shift 2
            ;;
        --) # end of options
            shift
            break
            ;;
        *)
            PYTEST_ARGS+=("$1")
            shift
            ;;
    esac
done

# Append any remaining args after --
if [[ $# -gt 0 ]]; then
    PYTEST_ARGS+=("$@")
fi

# Ensure baseline directory exists
mkdir -p "$BASELINE_DIR"

# Check pytest is available
if ! command -v pytest >/dev/null 2>&1; then
    echo "Error: pytest is not installed or not on PATH." >&2
    exit 1
fi

# Informational echo
#echo "Baseline directory: $BASELINE_DIR"
#echo "Mode: $MODE"
#echo "Pytest args: ${PYTEST_ARGS[*]:-<none>}"
#echo

# Run according to mode
if [[ "$MODE" == "generate" ]]; then
    # Generate/update baseline images
    # pytest-mpl expects --mpl and --mpl-generate-path
    pytest --mpl --mpl-generate-path "$BASELINE_DIR" "${PYTEST_ARGS[@]}"
else
    # Compare against existing baselines
    # You can set baseline path in pytest.ini, but passing here is explicit:
    pytest --mpl "${PYTEST_ARGS[@]}"
fi
