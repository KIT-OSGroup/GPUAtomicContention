#!/bin/bash

source config.sh

if [ "$NVIDIA" = "true" ]; then
    echo "Building for NVIDIA..."
    docker build . -f docker/nvidia.Dockerfile -t benchmarks/plos25-gpu-atomics-nvidia --build-arg nvidia_arch=$(cat nvidia_arch.txt) --progress=plain
fi

if [ "$AMD" = "true" ]; then
    echo "Building for AMD..."
    docker build . -f docker/amd.Dockerfile -t benchmarks/plos25-gpu-atomics-amd --build-arg amdgpu_arch=$(cat amdgpu_arch.txt) --progress=plain
fi
