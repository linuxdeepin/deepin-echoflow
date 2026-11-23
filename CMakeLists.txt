# SPDX-FileCopyrightText: 2026 Hualet Wang
# SPDX-License-Identifier: GPL-3.0-or-later

cmake_minimum_required(VERSION 3.16)
project(deepin-echoflow LANGUAGES C CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

option(ECHOFLOW_BUILD_TESTS "Build QTest unit tests + spec tests" ON)
set(ECHOFLOW_CPU_TARGET "native" CACHE STRING
    "CPU target for CrispASR: native or x86-64-v3")
set_property(CACHE ECHOFLOW_CPU_TARGET PROPERTY STRINGS native x86-64-v3)

if(ECHOFLOW_CPU_TARGET STREQUAL "x86-64-v3")
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
        message(FATAL_ERROR
            "x86-64-v3 requires an amd64 build host; use ECHOFLOW_CPU_TARGET=native "
            "for ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_SSE42 ON CACHE BOOL "" FORCE)
    set(GGML_AVX ON CACHE BOOL "" FORCE)
    set(GGML_AVX2 ON CACHE BOOL "" FORCE)
    set(GGML_BMI2 ON CACHE BOOL "" FORCE)
    set(GGML_FMA ON CACHE BOOL "" FORCE)
    set(GGML_F16C ON CACHE BOOL "" FORCE)
    set(GGML_AVX_VNNI OFF CACHE BOOL "" FORCE)
    set(GGML_AVX512 OFF CACHE BOOL "" FORCE)
    set(GGML_AVX512_VBMI OFF CACHE BOOL "" FORCE)
    set(GGML_AVX512_VNNI OFF CACHE BOOL "" FORCE)
    set(GGML_AVX512_BF16 OFF CACHE BOOL "" FORCE)
    set(GGML_AMX_TILE OFF CACHE BOOL "" FORCE)
    set(GGML_AMX_INT8 OFF CACHE BOOL "" FORCE)
    set(GGML_AMX_BF16 OFF CACHE BOOL "" FORCE)
elseif(NOT ECHOFLOW_CPU_TARGET STREQUAL "native"
       AND NOT ECHOFLOW_CPU_TARGET STREQUAL "x86-64-v3")
    message(FATAL_ERROR "Unsupported ECHOFLOW_CPU_TARGET=${ECHOFLOW_CPU_TARGET}")
endif()

# ggml enables the RISC-V vector extensions by default and compiles ggml-cpu
# with -march=..._zvfh, gated on `defined(__riscv_zvfh)` in the source. GCC 12
# and 13 — what riscv64 builds use on deepin 25 — both define the macro without
# the intrinsics the code needs: GCC 12 ships no riscv_vector.h and no _Float16
# for that target, and GCC 13 only has the v0.11 intrinsics while ggml uses the
# v1.0 API. binutils 2.41 additionally rejects the _zvfh suffix. Any of these
# breaks the CPU backend, so probe the toolchain with the code ggml-cpu uses and
# fall back to the scalar profile when it does not compile.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv64")
    include(CheckCSourceCompiles)
    set(_echoflow_try_compile_target_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
    set(_echoflow_required_flags "${CMAKE_REQUIRED_FLAGS}")
    # Compile only: the probe must exercise the assembler too, and a link into
    # a cross sysroot is not what decides whether ggml-cpu builds.
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -march=rv64gcv_zfh_zvfh -mabi=lp64d")
    check_c_source_compiles("
        #include <riscv_vector.h>
        int main(void) {
            vfloat32m2_t vx = __riscv_vle32_v_f32m2((const float *) 0, 1);
            vfloat16m1_t vy = __riscv_vfncvt_f_f_w_f16m1(vx, 1);
            __riscv_vse16_v_f16m1((_Float16 *) 0, vy, 1);
            return 0;
        }" ECHOFLOW_RISCV64_RVV_INTRINSICS)
    set(CMAKE_REQUIRED_FLAGS "${_echoflow_required_flags}")
    set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_echoflow_try_compile_target_type}")
    if(NOT ECHOFLOW_RISCV64_RVV_INTRINSICS)
        message(STATUS
            "riscv64 toolchain cannot compile the RISC-V vector intrinsics "
            "ggml-cpu uses; building ggml-cpu without them")
        set(GGML_RVV OFF CACHE BOOL "" FORCE)
    endif()
endif()

# Vendored CrispASR — build as static lib, link C-ABI into echoflow_service.
# Only the qwen3 backend is compiled (other backends stripped at configure time).
set(CRISPASR_WASM OFF CACHE BOOL "" FORCE)
set(CRISPASR_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CRISPASR_QWEN3_ONLY ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
# deepin-echoflow feeds live PCM (PipeWire) to the ASR engine — it never decodes
# .opus/.ogg files. Disabling opus skips FetchContent for libopus/libogg/
# opusfile and the miniaudio opus backend (~145 .o).
set(CRISPASR_OPUS OFF CACHE BOOL "" FORCE)
# deepin-echoflow never writes signed WAV files, so skip the C2PA signer and its
# nested submodule.
set(CRISPASR_NO_C2PA_NATIVE ON CACHE BOOL "" FORCE)

add_subdirectory(third_party/crispasr EXCLUDE_FROM_ALL)

# deepin-echoflow only consumes the qwen3 session API. Keep its C API dispatcher and
# the compatibility implementation that the dispatcher still references;
# omit file decoding, device I/O, VAD, diarization, alignment, enhancement,
# chat, and other library surfaces. Adjust the target after CrispASR declares
# it instead of rewriting submodule files, so no-op install-user.sh builds do
# not see changed source mtimes.
get_target_property(_crispasr_sources crispasr-lib SOURCES)
list(FILTER _crispasr_sources INCLUDE REGEX "(^|/)(crispasr|crispasr_c_api)\\.cpp$")
set_property(TARGET crispasr-lib PROPERTY SOURCES "${_crispasr_sources}")

# Link only the qwen3 runtime and its ggml/core dependencies. An allowlist
# keeps every other backend target unbuilt (EXCLUDE_FROM_ALL) without
# patching CrispASR's long, frequently changing backend link section.
set(_crispasr_link_allowlist
    "^(qwen3_asr|crisp_audio|crispasr-core|ggml|ggml-[a-z]+|Threads::Threads)$")
get_target_property(_crispasr_links crispasr-lib LINK_LIBRARIES)
list(FILTER _crispasr_links INCLUDE REGEX "${_crispasr_link_allowlist}")
set_property(TARGET crispasr-lib PROPERTY LINK_LIBRARIES "${_crispasr_links}")

get_target_property(_crispasr_interface_links crispasr-lib INTERFACE_LINK_LIBRARIES)
list(FILTER _crispasr_interface_links INCLUDE REGEX "${_crispasr_link_allowlist}")
set_property(TARGET crispasr-lib PROPERTY INTERFACE_LINK_LIBRARIES "${_crispasr_interface_links}")

target_compile_definitions(crispasr-lib PRIVATE CRISPASR_QWEN3_ONLY)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(crispasr-lib PRIVATE -ffunction-sections -fdata-sections)
endif()

# The optional transcript polish step runs a small chat model through
# CrispASR's text chat ABI. crispasr-lib above is trimmed to the ASR session
# sources, so build chat.cpp on its own against the vendored llama.cpp core.
add_library(echoflow_crisp_chat STATIC third_party/crispasr/src/chat.cpp)
target_include_directories(echoflow_crisp_chat PUBLIC
    ${CMAKE_SOURCE_DIR}/third_party/crispasr/include)
target_link_libraries(echoflow_crisp_chat PRIVATE crispasr-llama-core)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(echoflow_crisp_chat PRIVATE -ffunction-sections -fdata-sections)
    target_compile_options(crispasr-llama-core PRIVATE -ffunction-sections -fdata-sections)
endif()

# Keep the persistent RNNoise front-end separate from CrispASR's batch wrapper.
# These are unmodified vendored BSD sources; no model downloads are required.
find_package(PkgConfig REQUIRED)
pkg_check_modules(SAMPLERATE REQUIRED IMPORTED_TARGET samplerate)
add_library(echoflow_rnnoise STATIC
    third_party/crispasr/src/rnnoise/denoise.c
    third_party/crispasr/src/rnnoise/rnn.c
    third_party/crispasr/src/rnnoise/rnn_data.c
    third_party/crispasr/src/rnnoise/pitch.c
    third_party/crispasr/src/rnnoise/celt_lpc.c
    third_party/crispasr/src/rnnoise/kiss_fft.c)
target_include_directories(echoflow_rnnoise PUBLIC ${CMAKE_SOURCE_DIR}/third_party/crispasr/src/rnnoise)
target_link_libraries(echoflow_rnnoise PUBLIC m)

add_subdirectory(service)
add_subdirectory(fcitx-addon)
add_subdirectory(ui-host)
if(ECHOFLOW_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/systemd/user")
configure_file(
    "${CMAKE_SOURCE_DIR}/systemd/user/deepin-echoflow.service.in"
    "${CMAKE_BINARY_DIR}/systemd/user/deepin-echoflow.service"
    @ONLY)
configure_file(
    "${CMAKE_SOURCE_DIR}/systemd/user/deepin-echoflow.socket.in"
    "${CMAKE_BINARY_DIR}/systemd/user/deepin-echoflow.socket"
    @ONLY)
configure_file(
    "${CMAKE_SOURCE_DIR}/ui-host/deepin-echoflow-autostart.desktop.in"
    "${CMAKE_BINARY_DIR}/deepin-echoflow-autostart.desktop"
    @ONLY)
install(FILES
    "${CMAKE_BINARY_DIR}/systemd/user/deepin-echoflow.service"
    "${CMAKE_BINARY_DIR}/systemd/user/deepin-echoflow.socket"
    DESTINATION "lib/systemd/user")
install(FILES "${CMAKE_BINARY_DIR}/deepin-echoflow-autostart.desktop"
    DESTINATION "${CMAKE_INSTALL_SYSCONFDIR}/xdg/autostart")
install(PROGRAMS
    "${CMAKE_SOURCE_DIR}/scripts/deepin-echoflow-session-start"
    "${CMAKE_SOURCE_DIR}/scripts/deepin-echoflow-session-stop"
    DESTINATION "${CMAKE_INSTALL_BINDIR}")
