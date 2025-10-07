#!/bin/bash

source config.sh

IFS=',' read -ra nvidia_devices <<< "$NVIDIA_DEVICES"

truncate --size=0 nvidia_arch.txt

if [ "$NVIDIA_DEVICES" = "all" ]; then
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed -e 's/\.//g' >> nvidia_arch.txt
else
    for device in ${nvidia_devices}; do
        nvidia-smi --query-gpu=compute_cap --format=csv,noheader --id=$device | sed -e 's/\.//g' >> nvidia_arch.txt
    done
fi

cat nvidia_arch.txt | sort -u | awk '{print "-gencode=arch=compute_" $0 ",code=sm_" $0}' | sed -e ':a;N;$!ba;s/\n/ /g' > nvidia_arch.txt
