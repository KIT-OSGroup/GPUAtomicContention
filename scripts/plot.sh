#!/bin/bash

source config.sh

IFS=',' read -ra ARR_AMD_FOLDERS <<< "$AMD_FOLDERS"
IFS=',' read -ra ARR_NVIDIA_FOLDERS <<< "$NVIDIA_FOLDERS"
ARR_FOLDERS+=( "${ARR_AMD_FOLDERS[@]}" "${ARR_NVIDIA_FOLDERS[@]}" )
GPU_STRING=$(IFS=',' ; echo "${ARR_FOLDERS[*]}")

python3 ./scripts/paper_plots.py ./data $GPU_STRING
