FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

# Install build-essentials
RUN export DEBIAN_FRONTEND=noninteractive; \
    apt-get update && apt-get install --no-install-recommends -y \
    gnupg \
    build-essential \
    cmake \
    libgtest-dev \
    libbenchmark-dev \
    curl \
    libfile-basedir-perl \
    && rm -rf /var/lib/apt/lists/*

# Install HIP development environment
RUN export DEBIAN_FRONTEND=noninteractive; \
    curl https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - \
    && echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.1.1 jammy main" > /etc/apt/sources.list.d/rocm.list \
    && apt-get update \
    && apt-get install -y hip-base hipify-clang rocm-llvm \
    && apt-get download hip-runtime-nvidia hip-dev hipcc \
    && dpkg -i hip*

# Install NVIDIA Nsight
RUN export DEBIAN_FRONTEND=noninteractive; \
    echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu2204/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list \
    && apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub \
    && apt-get update \
    && apt-get install -y nsight-systems-cli

# Setup environment
ENV HIP_PLATFORM=nvidia
ARG nvidia_arch
ENV NVIDIA_ARCH=${nvidia_arch:-"-gencode arch=compute_86,code=sm_86"}
ENV PATH="/opt/rocm/bin:${PATH}"
RUN echo "/opt/rocm/lib" >> /etc/ld.so.conf.d/rocm.conf && ldconfig

# Build benchmarks
COPY benchmarks /benchmarks
RUN mkdir /benchmarks/executables
WORKDIR /benchmarks/build
RUN cmake -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=/benchmarks/executables ..
RUN make -j 16

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS runtime

RUN export DEBIAN_FRONTEND=noninteractive; \
    apt-get update && apt-get install --no-install-recommends -y \
    curl \
    libfile-basedir-perl \
    libgtest-dev \
    libbenchmark-dev \
    gdb \
    bsdmainutils \
    && rm -rf /var/lib/apt/lists/*

# Install HIP runtime environment
RUN export DEBIAN_FRONTEND=noninteractive; \
    curl https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - \
    && echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.1.1 jammy main" > /etc/apt/sources.list.d/rocm.list \
    && apt-get update \
    && apt-get install -y hip-base hipify-clang rocm-llvm \
    && apt-get download hip-runtime-nvidia hip-dev hipcc \
    && dpkg -i hip*

# Install NVIDIA Nsight
RUN export DEBIAN_FRONTEND=noninteractive; \
    echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu2204/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list \
    && apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub \
    && apt-get update \
    && apt-get install -y nsight-systems-cli

COPY --from=builder /benchmarks/executables/* /bin/

# Copy sources for debugging
COPY benchmarks /benchmarks
COPY --from=builder /benchmarks/build/compile_commands.json /
