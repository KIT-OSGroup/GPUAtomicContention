#!/bin/bash

source config.sh

UID=$(id -u)
GID=$(id -g)
CMDLINE=$@

IFS=',' read -ra ARR_AMD_DEVICES <<< "$AMD_DEVICES"
IFS=',' read -ra ARR_AMD_FOLDERS <<< "$AMD_FOLDERS"

for i in "${!ARR_AMD_DEVICES[@]}"; do
    mkdir -p data/"${ARR_AMD_FOLDERS[i]}"

    docker run --rm -iv ${PWD}/data/"${ARR_AMD_FOLDERS[i]}":/out --privileged --device /dev/kfd --device ${ARR_AMD_DEVICES[i]} --security-opt seccomp=unconfined benchmarks/plos25-gpu-atomics-amd sh -s <<EOF
$CMDLINE
chown -R $UID:$GID /out
EOF
done
