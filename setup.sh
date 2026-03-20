#!/usr/bin/env bash

# --------------------------------------------------
# ECE476 environment setup
#
# Correct usage:
#   source setup.sh
#
# This file must be sourced, not executed.
# --------------------------------------------------

_ece476_quiet=0
for arg in "$@"; do
    case "$arg" in
        -q|--quiet)
            _ece476_quiet=1
            ;;
    esac
done

# Detect whether this file is being sourced.
# Works in bash and zsh.
_ece476_is_sourced=0
if [ -n "${ZSH_EVAL_CONTEXT:-}" ]; then
    case $ZSH_EVAL_CONTEXT in
        *:file) _ece476_is_sourced=1 ;;
    esac
elif [ -n "${BASH_VERSION:-}" ]; then
    if [ "${BASH_SOURCE[0]}" != "$0" ]; then
        _ece476_is_sourced=1
    fi
fi

if [ "$_ece476_is_sourced" -ne 1 ]; then
    echo "ERROR: This script must be sourced, not executed."
    echo "Use:"
    echo "  source setup.sh"
    echo "or:"
    echo "  . setup.sh"
    exit 1
fi

unset _ece476_is_sourced

# Guard against repeated sourcing
if [ -n "${ECE476_ENV_LOADED:-}" ]; then
    if [ "$_ece476_quiet" -ne 1 ]; then
        echo "[ECE476] Environment already loaded. Skipping."
    fi
    return 0
fi

# Add course module path once
case ":${MODULEPATH:-}:" in
    *:/scratch/network/zl1111/ece476-public/modulefiles:*) ;;
    *) export MODULEPATH=/scratch/network/zl1111/ece476-public/modulefiles:${MODULEPATH:-} ;;
esac

# Load required modules
module load gcc/15.2.0
module load freeglut/3.6.0
module load cudatoolkit/13.0

# Default Slurm settings
_ece476_srun_defaults=(
    -p gpu
    --gres=gpu:3g.20gb:1
    --time=00:15:00
    --cpus-per-task=6
    --mem=8G
)

ece476-srun() {
    if [ "$#" -eq 0 ]; then
        echo "Usage: ece476-srun [srun options] <command> [args...]"
        echo "Examples:"
        echo "  ece476-srun ./cudaSaxpy"
        echo "  ece476-srun --time=00:20:00 ./cudaSaxpy"
        echo "  ece476-srun --mem=12G --cpus-per-task=8 ./cudaSaxpy"
        return 1
    fi

    srun "${_ece476_srun_defaults[@]}" "$@"
}


if [ "$_ece476_quiet" -ne 1 ]; then
    echo "[ECE476] Environment loaded successfully."
    echo "[ECE476] Use: ece476-srun <command>"
fi

# avoid reloading environment in the same shell
ECE476_ENV_LOADED=1
