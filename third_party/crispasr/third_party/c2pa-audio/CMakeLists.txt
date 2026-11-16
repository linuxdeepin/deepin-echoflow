cmake_minimum_required(VERSION 3.14)
project(c2pa_audio VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

set(_c2pa_sources src/capi.cpp src/c2pa_native.cpp third_party/uecc/uECC.c)

# Shared library — what the FFI bindings (Dart/C#/Python/Go) load.
add_library(c2pa_audio SHARED ${_c2pa_sources})
target_include_directories(c2pa_audio PUBLIC include PRIVATE src third_party/uecc)

# Static library — for consumers that link at build time (cgo, C/C++).
add_library(c2pa_audio_static STATIC ${_c2pa_sources})
target_include_directories(c2pa_audio_static PUBLIC include PRIVATE src third_party/uecc)
set_target_properties(c2pa_audio_static PROPERTIES OUTPUT_NAME c2pa_audio)

# Silence vendored micro-ecc warnings.
if (NOT MSVC)
    set_source_files_properties(third_party/uecc/uECC.c PROPERTIES COMPILE_OPTIONS "-w")
endif()

# Tiny C smoke test (sign -> verify round-trip through the C ABI).
option(C2PA_AUDIO_BUILD_TESTS "Build the C ABI smoke test" ON)
if (C2PA_AUDIO_BUILD_TESTS)
    enable_testing()
    add_executable(c2pa_capi_test test/c2pa_capi_test.c)
    target_link_libraries(c2pa_capi_test PRIVATE c2pa_audio_static)
    target_compile_definitions(c2pa_capi_test PRIVATE
        C2PA_AUDIO_TEST_ASSETS="${CMAKE_CURRENT_SOURCE_DIR}/test/assets")
    add_test(NAME c2pa_capi_test COMMAND c2pa_capi_test)
endif()

# Opt-in mutation fuzzer for the verifier (untrusted-input parser). Builds the
# core + fuzz harness with AddressSanitizer + UndefinedBehaviorSanitizer.
#   cmake -B build-fuzz -DC2PA_AUDIO_FUZZ=ON && cmake --build build-fuzz
#   ./build-fuzz/c2pa_fuzz_verify 200000 test/assets/reference-c2pa-rs.*
option(C2PA_AUDIO_FUZZ "Build the ASan/UBSan verifier fuzzer" OFF)
if (C2PA_AUDIO_FUZZ)
    add_executable(c2pa_fuzz_verify test/fuzz_verify.cpp src/c2pa_native.cpp third_party/uecc/uECC.c)
    target_include_directories(c2pa_fuzz_verify PRIVATE src third_party/uecc)
    target_compile_options(c2pa_fuzz_verify PRIVATE -g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(c2pa_fuzz_verify PRIVATE -fsanitize=address,undefined)
    if (NOT MSVC)
        set_source_files_properties(third_party/uecc/uECC.c PROPERTIES COMPILE_OPTIONS "-w")
    endif()
endif()

install(TARGETS c2pa_audio c2pa_audio_static
        LIBRARY DESTINATION lib ARCHIVE DESTINATION lib RUNTIME DESTINATION bin)
install(FILES include/c2pa_audio.h DESTINATION include)
