FROM nvcr.io/nvidia/pytorch:24.02-py3

ARG NCCL_BRANCH=v2.19
ARG REPO=https://github.com/NVIDIA/nccl.git
ARG VERSION=2.21.5-1

RUN apt-get update && apt-get install -y --no-install-recommends \
        autotools-dev \
        automake \
        autoconf \
        git \
        libtool \
		python3 \
		moreutils \
        flex && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN mkdir -p /workspace/nccl/build/lib && \
    cp /lib/x86_64-linux-gnu/libnccl.so.* /workspace/nccl/build/lib/

RUN cd /opt && \
    git clone -b moe-benchmarks https://github.com/x41lakazam/nccl-tests && \
    cd nccl-tests && \
    make BUILDDIR="/usr/local/bin" MPI=1 MPI_HOME=/usr/local/mpi CUDA_HOME=/usr/local/cuda NCCL_HOME=/workspace/nccl/build

RUN mv /usr/local/bin/bisection_perf /usr/local/bin/bisection_perf_mpi
RUN mv /usr/local/bin/moe_benchmark_perf /usr/local/bin/moe_benchmark_perf_mpi

RUN rm -rf /workspace/nccl/

RUN apt-get remove -y \
        autotools-dev \
        automake \
        autoconf \
        git \
        libtool \
        flex

