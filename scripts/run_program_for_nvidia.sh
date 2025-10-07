#!/bin/bash

source config.sh

UID=$(id -u)
GID=$(id -g)
CMDLINE=$@

IFS=',' read -ra ARR_NVIDIA_DEVICES <<< "$NVIDIA_DEVICES"
IFS=',' read -ra ARR_NVIDIA_FOLDERS <<< "$NVIDIA_FOLDERS"

for i in "${!ARR_NVIDIA_DEVICES[@]}"; do
    mkdir -p data/"${ARR_NVIDIA_FOLDERS[i]}"

    docker run --rm -iv ${PWD}/data/"${ARR_NVIDIA_FOLDERS[i]}":/out --privileged --gpus device=${ARR_NVIDIA_DEVICES[i]} --security-opt seccomp=unconfined benchmarks/plos25-gpu-atomics-nvidia sh -s <<EOF
nsys profile $CMDLINE
chown -R $UID:$GID /out
EOF
done
