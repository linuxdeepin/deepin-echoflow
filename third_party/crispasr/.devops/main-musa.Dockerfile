ARG UBUNTU_VERSION=22.04
# This needs to generally match the container host's environment.
ARG MUSA_VERSION=rc4.2.0
# Target the MUSA build image
ARG BASE_MUSA_DEV_CONTAINER=mthreads/musa:${MUSA_VERSION}-devel-ubuntu${UBUNTU_VERSION}-amd64
# Target the MUSA runtime image
ARG BASE_MUSA_RUN_CONTAINER=mthreads/musa:${MUSA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}-amd64

FROM ${BASE_MUSA_DEV_CONTAINER} AS build
WORKDIR /app

RUN apt-get update && \
    apt-get install -y build-essential libsdl2-dev wget cmake git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /tmp/* /var/tmp/*

COPY . .
# Enable muBLAS
# NOTE (#261): GGML_NATIVE=OFF does NOT restrict the ISA to the flags listed —
# ggml sets INS_ENB=ON in that case, so BMI2/SSE42/AVX default ON anyway. The
# real baseline is Haswell-class (AVX2+FMA+F16C+BMI2); spelled out explicitly
# so it's visible. See .devops/main-cuda.Dockerfile for the full explanation.
RUN cmake -B build -DCRISPASR_BUILD_TESTS=OFF -DGGML_MUSA=1 \
    -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
    -DGGML_BMI2=ON -DGGML_SSE42=ON -DGGML_AVX=ON -DGGML_AVX512=OFF && \
    cmake --build build -j"$(nproc)" --target crispasr-cli

RUN find /app/build -name "*.o" -delete && \
    find /app/build -name "*.a" -delete && \
    rm -rf /app/build/CMakeFiles && \
    rm -rf /app/build/cmake_install.cmake && \
    rm -rf /app/build/_deps

FROM ${BASE_MUSA_RUN_CONTAINER} AS runtime
WORKDIR /app

RUN apt-get update && \
    apt-get install -y curl passwd ffmpeg wget cmake git tini && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /tmp/* /var/tmp/*

COPY --from=build /app/build/bin /app/build/bin
COPY --from=build /app/samples /app/samples
# /models is a volume mount point — actual models (and the convert-*.py
# tools) are excluded by .dockerignore. Don't COPY them from the build
# stage; the user mounts their model directory at /models at runtime.
RUN (id -u crispasr 2>/dev/null || \
     useradd -m -u 1000 crispasr 2>/dev/null || \
     useradd -m crispasr) && \
    mkdir -p /cache /models && \
    chown -R crispasr:crispasr /app /cache /models

ENV PATH=/app/build/bin:$PATH
USER crispasr
ENTRYPOINT [ "tini", "--", "bash", "/app/.devops/run-server.sh" ]
