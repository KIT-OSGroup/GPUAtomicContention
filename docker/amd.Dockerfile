FROM rocm/dev-ubuntu-22.04:6.2.4-complete AS builder

# Install build-essentials
RUN export DEBIAN_FRONTEND=noninteractive; \
    apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    cmake \
    libgtest-dev \
    libbenchmark-dev \
    curl \
    libfile-basedir-perl \
    ltrace \
    vim \
    less \
    && rm -rf /var/lib/apt/lists/*

# Setup environment
ENV HIP_PLATFORM=amd
ARG amdgpu_arch
ENV AMDGPU_ARCH=${amdgpu_arch:-"--offload-arch=gfx1030"}

# Build benchmarks
COPY benchmarks /benchmarks
RUN mkdir /benchmarks/executables
WORKDIR /benchmarks/build
RUN cmake -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=/benchmarks/executables ..
RUN make -j 16

FROM rocm/dev-ubuntu-22.04:6.2.4-complete AS runtime

RUN export DEBIAN_FRONTEND=noninteractive; \
    apt-get update && apt-get install --no-install-recommends -y \
    curl \
    libfile-basedir-perl \
    libgtest-dev \
    libbenchmark-dev \
    gdb \
    bsdmainutils \
    ltrace \
    vim \
    less \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /benchmarks/executables/* /bin/

# Copy sources for debugging
COPY benchmarks /benchmarks
COPY --from=builder /benchmarks/build/compile_commands.json /
