# crisp_audio — shared C++ audio-encoder library.
#
# Used by CrispASR (qwen3_asr.cpp) and CrispEmbed (BidirLM-Omni audio path)
# to share the Whisper-shape Conv-stem + Transformer-encoder forward pass.
#
# Linked into CrispEmbed via add_subdirectory("../CrispASR/crisp_audio") from
# CrispEmbed's top-level CMakeLists.txt.

cmake_minimum_required(VERSION 3.14)

# crisp_audio links against the consumer's "core" target — which provides
# core_gguf::* and core_mel::* symbols. CrispASR's crispasr-core and
# CrispEmbed's crispembed-core both supply these from identical sources;
# we pick whichever the parent build defines so consumers don't end up
# with duplicate symbols.
if(TARGET crispasr-core)
    set(_crisp_audio_core_target crispasr-core)
    set(_crisp_audio_core_inc    ${CMAKE_CURRENT_SOURCE_DIR}/../src)
elseif(TARGET crispembed-core)
    set(_crisp_audio_core_target crispembed-core)
    # CrispEmbed exposes core/* via crispembed-core's PUBLIC include dirs;
    # nothing extra needed here.
    set(_crisp_audio_core_inc "")
else()
    message(STATUS "crisp_audio: no core target (crispasr-core or crispembed-core) — skipping")
    return()
endif()

add_library(crisp_audio STATIC
    src/audio_tower.cpp
    include/crisp_audio.h
)

target_include_directories(crisp_audio
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
if(_crisp_audio_core_inc)
    target_include_directories(crisp_audio PRIVATE ${_crisp_audio_core_inc})
endif()

target_link_libraries(crisp_audio
    PUBLIC  ${_crisp_audio_core_target} ggml ggml-base
)

target_compile_features(crisp_audio PUBLIC cxx_std_17)
set_target_properties(crisp_audio PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(NOT MSVC)
    target_compile_options(crisp_audio PRIVATE -O3)
endif()

# GPU backend linkage.
#
# `crispasr_link_ggml_{metal,cuda}` are INTERFACE targets defined by CrispASR's
# TOP-LEVEL CMakeLists (the #355 GGML_BACKEND_DL work: empty under DL, the real
# backend otherwise). They do NOT exist when this directory is
# add_subdirectory()'d into CrispEmbed — and CMake does not error on an unknown
# name here, it degrades it to a raw `-lcrispasr_link_ggml_cuda`, so the failure
# surfaces at LINK time as:
#
#     mold: library not found: crispasr_link_ggml_cuda
#
# That is CrispEmbed issue #50's defect class again, and it hid longer than the
# others because it needs the sibling layout AND a GPU build: every CPU sibling
# build passes, including the sibling-crispasr CI job, since GitHub runners have
# no CUDA. It was caught on a Kaggle P100.
#
# So probe for the target rather than assuming the consumer defines it, and fall
# back to the plain backend target. The `NOT GGML_BACKEND_DL` guard on the
# fallback matters: under DL the backends are MODULE libraries, which CMake
# refuses as link targets and which must not be linked anyway — a missing
# backend is meant to be a runtime no-op, not a load-time abort.
if(GGML_METAL)
    target_compile_definitions(crisp_audio PRIVATE GGML_USE_METAL)
    if(TARGET crispasr_link_ggml_metal)
        target_link_libraries(crisp_audio PUBLIC crispasr_link_ggml_metal)
    elseif(TARGET ggml-metal AND NOT GGML_BACKEND_DL)
        target_link_libraries(crisp_audio PUBLIC ggml-metal)
    endif()
endif()
if(GGML_CUDA)
    target_compile_definitions(crisp_audio PRIVATE GGML_USE_CUDA)
    if(TARGET crispasr_link_ggml_cuda)
        target_link_libraries(crisp_audio PUBLIC crispasr_link_ggml_cuda)
    elseif(TARGET ggml-cuda AND NOT GGML_BACKEND_DL)
        target_link_libraries(crisp_audio PUBLIC ggml-cuda)
    endif()
endif()
