.PHONY: all
all: run-nvidia run-amd
	./scripts/plot.sh

## Plotting

.PHONY: plot
plot:
	./scripts/plot.sh

## Running

.PHONY: run-nvidia
run-nvidia: build
	./scripts/runner.sh ./scripts/run_program_for_nvidia.sh

.PHONY: run-amd
run-amd: build
	./scripts/runner.sh ./scripts/run_program_for_amd.sh

.PHONY: enter-nvidia
enter-nvidia: build
	docker run --rm -itv /tmp:/store --privileged --entrypoint /bin/bash --gpus all --security-opt seccomp=unconfined benchmarks/plos25-gpu-atomics-nvidia

.PHONY: enter-amd
enter-amd: build
	docker run --rm -itv /tmp:/store --privileged --entrypoint /bin/bash --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined benchmarks/plos25-gpu-atomics-amd

## Building

.PHONY: build
build: nvidia_arch.txt amdgpu_arch.txt
	./scripts/build.sh

nvidia_arch.txt:
	./scripts/generate_nvidia_arch.sh

amdgpu_arch.txt:
	amdgpu-arch | sort -u | sed -e 's/^/--offload-arch=/' | sed -e ':a;N;$!ba;s/\n/ /g' > amdgpu_arch.txt

## Cleaning

.PHONY: clean
clean: clean-data clean-plots clean-arch-files

.PHONY: clean-data
clean-data:
	rm -rf data

.PHONY: clean-plots
clean-plots:
	rm -rf plots

.PHONY: clean-arch-files
clean-arch-files:
	rm -f amdgpu_arch.txt
	rm -f nvidia_arch.txt
