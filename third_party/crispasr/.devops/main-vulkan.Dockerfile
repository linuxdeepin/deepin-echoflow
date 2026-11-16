FROM ubuntu:24.04 AS build
WORKDIR /app

RUN apt-get update && \
  apt-get install -y build-essential wget cmake git libvulkan-dev glslc spirv-headers \
  && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY . .
# NOTE (#261): GGML_NATIVE=OFF does NOT restrict the ISA to the flags listed —
# ggml sets INS_ENB=ON in that case, so BMI2/SSE42/AVX default ON anyway. The
# real baseline is Haswell-class (AVX2+FMA+F16C+BMI2); spelled out explicitly
# so it's visible. See .devops/main-cuda.Dockerfile for the full explanation.
RUN cmake -B build -DCRISPASR_BUILD_TESTS=OFF -DGGML_VULKAN=1 \
    -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
    -DGGML_BMI2=ON -DGGML_SSE42=ON -DGGML_AVX=ON -DGGML_AVX512=OFF && \
  cmake --build build -j"$(nproc)" --target crispasr-cli

FROM ubuntu:24.04 AS runtime
WORKDIR /app

RUN apt-get update && \
  apt-get install -y curl passwd ffmpeg libsdl2-dev wget cmake git tini libvulkan1 mesa-vulkan-drivers \
  && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY --from=build /app /app
RUN (id -u crispasr 2>/dev/null || \
     useradd -m -u 1000 crispasr 2>/dev/null || \
     useradd -m crispasr) && \
    mkdir -p /cache /models && \
    chown -R crispasr:crispasr /app /cache /models
ENV PATH=/app/build/bin:$PATH
USER crispasr
ENTRYPOINT [ "tini", "--", "bash", "/app/.devops/run-server.sh" ]
