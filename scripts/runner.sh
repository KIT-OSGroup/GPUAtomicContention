#!/bin/bash

source config.sh

RUNNER=$1

if [ "$BM_CONTENTION" = "true" ]; then
    BM_CONTENTION_CMDLINE="bm_contention"

    if [ -n "$BM_CONTENTION_MODES" ]; then
        BM_CONTENTION_CMDLINE="$BM_CONTENTION_CMDLINE --modes=$BM_CONTENTION_MODES"
    fi

    if [ -n "$BM_CONTENTION_SCOPES" ]; then
        BM_CONTENTION_CMDLINE="$BM_CONTENTION_CMDLINE --scopes=$BM_CONTENTION_SCOPES"
    fi

    if [ -n "$BM_CONTENTION_GRIDS" ]; then
        BM_CONTENTION_CMDLINE="$BM_CONTENTION_CMDLINE --grids=$BM_CONTENTION_GRIDS"
    fi

    if [ -n "$BM_CONTENTION_TRANSPOSE_MODES" ]; then
        BM_CONTENTION_CMDLINE="$BM_CONTENTION_CMDLINE --transpose=$BM_CONTENTION_TRANSPOSE_MODES"
    fi

    if [ -n "$BM_CONTENTION_RUNS" ]; then
        BM_CONTENTION_CMDLINE="$BM_CONTENTION_CMDLINE --runs=$BM_CONTENTION_RUNS"
    fi

    if [ -n "$BM_CONTENTION_BASELINE_RUNS" ]; then
        BM_CONTENTION_CMDLINE="$BM_CONTENTION_CMDLINE --baseline-runs=$BM_CONTENTION_BASELINE_RUNS"
    fi

    if [ "$BM_CONTENTION_NO_BASELINE" = true ]; then
        BM_CONTENTION_CMDLINE="$BM_CONTENTION_CMDLINE --no-baseline"
    fi

    if [ "$BM_CONTENTION_NO_VARYING_THREADS" = true ]; then
        BM_CONTENTION_CMDLINE="$BM_CONTENTION_CMDLINE --no-varying-threads"
    fi

    if [ "$BM_CONTENTION_NO_VARYING_STRIDES" = true ]; then
        BM_CONTENTION_CMDLINE="$BM_CONTENTION_CMDLINE --no-varying-strides"
    fi

    if [ "$BM_CONTENTION_NO_OFFSETTED_STRIDES" = true ]; then
        BM_CONTENTION_CMDLINE="$BM_CONTENTION_CMDLINE --no-offsetted-strides"
    fi

    echo "Running '$BM_CONTENTION_CMDLINE'"
    $RUNNER $BM_CONTENTION_CMDLINE
else
    echo "Skipping 'bm_contention'!"
fi

if [ "$BM_ATOMICS" = "true" ]; then
    BM_ATOMICS_CMDLINE="bm_atomics"

    if [ -n "$BM_ATOMICS_VARIABLE_SPINLOCK_RUNS" ]; then
        BM_ATOMICS_CMDLINE="$BM_ATOMICS_CMDLINE --variable-spinlock-runs=$BM_ATOMICS_VARIABLE_SPINLOCK_RUNS"
    fi

    if [ -n "$BM_ATOMICS_VARIABLE_SPINLOCK_MODES" ]; then
        BM_ATOMICS_CMDLINE="$BM_ATOMICS_CMDLINE --variable-spinlock-modes=$BM_ATOMICS_VARIABLE_SPINLOCK_MODES"
    fi

    if [ -n "$BM_ATOMICS_STORE_RUNS" ]; then
        BM_ATOMICS_CMDLINE="$BM_ATOMICS_CMDLINE --store-runs=$BM_ATOMICS_STORE_RUNS"
    fi

    if [ -n "$BM_ATOMICS_STORE_MODES" ]; then
        BM_ATOMICS_CMDLINE="$BM_ATOMICS_CMDLINE --store-modes=$BM_ATOMICS_STORE_MODES"
    fi

    if [ -n "$BM_ATOMICS_FIXED_SPINLOCK_RUNS" ]; then
        BM_ATOMICS_CMDLINE="$BM_ATOMICS_CMDLINE --fixed-spinlock-runs=$BM_ATOMICS_FIXED_SPINLOCK_RUNS"
    fi

    if [ -n "$BM_ATOMICS_FIXED_SPINLOCK_MODES" ]; then
        BM_ATOMICS_CMDLINE="$BM_ATOMICS_CMDLINE --fixed-spinlock-modes=$BM_ATOMICS_FIXED_SPINLOCK_MODES"
    fi

    if [ "$BM_ATOMICS_NO_VARIABLE_SPINLOCK" = true ]; then
        BM_ATOMICS_CMDLINE="$BM_ATOMICS_CMDLINE --no-variable-spinlock"
    fi

    if [ "$BM_ATOMICS_NO_STORE" = true ]; then
        BM_ATOMICS_CMDLINE="$BM_ATOMICS_CMDLINE --no-store"
    fi

    if [ "$BM_ATOMICS_NO_FIXED_SPINLOCK" = true ]; then
        BM_ATOMICS_CMDLINE="$BM_ATOMICS_CMDLINE --no-fixed-spinlock"
    fi

    echo "Running '$BM_ATOMICS_CMDLINE'"
    $RUNNER $BM_ATOMICS_CMDLINE
else
    echo "Skipping 'bm_atomics'!"
fi
