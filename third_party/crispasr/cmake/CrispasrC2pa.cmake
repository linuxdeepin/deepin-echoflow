# CrispasrC2pa.cmake — shared C2PA (Content Credentials) enablement.
#
# Include once from the top-level CMakeLists; then call crispasr_enable_c2pa(<tgt>)
# for every target that includes core/crispasr_c2pa.h (the core lib + the CLI, so
# C2PA reaches the C API / wasm / bindings / server, not just the CLI binary).
#
# The prebuilt c2pa-rs native lib is `libc2pa_c`; scripts/fetch-c2pa.sh drops it
# under third_party/c2pa, or -DCRISPASR_C2PA_FETCH=ON downloads it here (below).

if (DEFINED _CRISPASR_C2PA_INCLUDED)
    return()
endif()
set(_CRISPASR_C2PA_INCLUDED ON)

set(_c2pa_hint "${CMAKE_SOURCE_DIR}/third_party/c2pa")

# -DCRISPASR_C2PA_FETCH=ON downloads the prebuilt native lib at configure time,
# cross-platform via CMake's own HTTP + unzip (no bash — works in CI on
# Linux/macOS/Windows). Best-effort: a failed download leaves C2PA disabled.
option(CRISPASR_C2PA_FETCH "Download the prebuilt c2pa-rs native lib if absent" OFF)
set(CRISPASR_C2PA_VERSION "0.89.3" CACHE STRING "c2pa-rs prebuilt version to fetch")
if (CRISPASR_C2PA_FETCH AND NOT EXISTS "${_c2pa_hint}/include/c2pa.h")
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _c2pa_arch)
    if (_c2pa_arch MATCHES "arm64|aarch64")
        set(_c2pa_cpu "aarch64")
    elseif (_c2pa_arch MATCHES "armv7|armeabi")
        set(_c2pa_cpu "armv7")
    else()
        set(_c2pa_cpu "x86_64")
    endif()
    # c2pa-rs ships a prebuilt for every CrispASR target: desktop
    # (dynamic .so/.dylib/.dll), Android (.so), iOS (static .a), and
    # wasm32-emscripten (static .a + .wasm). Static targets link straight in.
    if (EMSCRIPTEN)
        set(_c2pa_triple "wasm32-unknown-emscripten")
    elseif (ANDROID)
        if (_c2pa_cpu STREQUAL "aarch64")
            set(_c2pa_triple "aarch64-linux-android")
        elseif (_c2pa_cpu STREQUAL "armv7")
            set(_c2pa_triple "armv7-linux-androideabi")
        else()
            set(_c2pa_triple "x86_64-linux-android")
        endif()
    elseif (CMAKE_SYSTEM_NAME STREQUAL "iOS")
        set(_c2pa_triple "${_c2pa_cpu}-apple-ios")
    elseif (APPLE)
        set(_c2pa_triple "${_c2pa_cpu}-apple-darwin")
    elseif (WIN32)
        set(_c2pa_triple "${_c2pa_cpu}-pc-windows-msvc")
    else()
        set(_c2pa_triple "${_c2pa_cpu}-unknown-linux-gnu")
    endif()
    set(_c2pa_url
        "https://github.com/contentauth/c2pa-rs/releases/download/c2pa-v${CRISPASR_C2PA_VERSION}/c2pa-v${CRISPASR_C2PA_VERSION}-${_c2pa_triple}.zip")
    message(STATUS "C2PA: fetching ${_c2pa_url}")
    file(DOWNLOAD "${_c2pa_url}" "${CMAKE_BINARY_DIR}/c2pa-prebuilt.zip" STATUS _c2pa_dl TIMEOUT 120)
    list(GET _c2pa_dl 0 _c2pa_dl_code)
    if (_c2pa_dl_code EQUAL 0)
        file(MAKE_DIRECTORY "${_c2pa_hint}")
        file(ARCHIVE_EXTRACT INPUT "${CMAKE_BINARY_DIR}/c2pa-prebuilt.zip" DESTINATION "${_c2pa_hint}")
    else()
        message(WARNING "C2PA: prebuilt download failed (${_c2pa_dl}); signing will be disabled")
    endif()
endif()

find_library(C2PA_LIBRARY NAMES c2pa_c c2pa HINTS "${_c2pa_hint}/lib" NO_CMAKE_FIND_ROOT_PATH)
find_path(C2PA_INCLUDE_DIR NAMES c2pa.h c2pa/c2pa.h HINTS "${_c2pa_hint}/include" NO_CMAKE_FIND_ROOT_PATH)

set(CRISPASR_C2PA_STATIC OFF)
if (C2PA_LIBRARY)
    get_filename_component(_c2pa_ext "${C2PA_LIBRARY}" LAST_EXT)
    if (_c2pa_ext STREQUAL ".a")
        set(CRISPASR_C2PA_STATIC ON)
    endif()
endif()

if (C2PA_LIBRARY AND C2PA_INCLUDE_DIR)
    set(CRISPASR_C2PA_ENABLED ON)
    if (CRISPASR_C2PA_STATIC)
        message(STATUS "C2PA signing enabled — static (${C2PA_LIBRARY})")
    else()
        message(STATUS "C2PA signing enabled — dynamic sidecar (${C2PA_LIBRARY})")
    endif()
else()
    set(CRISPASR_C2PA_ENABLED OFF)
    message(STATUS "C2PA signing disabled (run scripts/fetch-c2pa.sh or pass -DCRISPASR_C2PA_FETCH=ON)")
endif()

# Apply C2PA to a target: link the lib, add the include + define, and (for a
# dynamic lib) an origin-relative rpath so a bundled sidecar resolves next to the
# binary. No-op when C2PA is unavailable. Safe to call for multiple targets.
function(crispasr_enable_c2pa TARGET)
    if (NOT CRISPASR_C2PA_ENABLED)
        return()
    endif()
    target_link_libraries(${TARGET} PRIVATE ${C2PA_LIBRARY})
    target_include_directories(${TARGET} PRIVATE ${C2PA_INCLUDE_DIR})
    target_compile_definitions(${TARGET} PRIVATE CRISPASR_HAVE_C2PA=1)
    # On Apple platforms the STATIC c2pa lib (iOS) pulls in c2pa-rs's native TLS /
    # keychain deps, so the consuming target must link the system frameworks or
    # the app link fails with undefined _SecTrust*/_CF*/_SCDynamicStore* symbols.
    # (The macOS DYNAMIC .dylib links these itself, so only guard the static case.)
    if (CRISPASR_C2PA_STATIC AND APPLE)
        target_link_libraries(${TARGET} PRIVATE "-framework Security" "-framework CoreFoundation"
                                                "-framework SystemConfiguration")
    endif()
    if (NOT CRISPASR_C2PA_STATIC)
        if (APPLE)
            set_property(TARGET ${TARGET} APPEND PROPERTY BUILD_RPATH "@loader_path")
            set_property(TARGET ${TARGET} APPEND PROPERTY INSTALL_RPATH "@loader_path")
        elseif (UNIX AND NOT EMSCRIPTEN)
            set_property(TARGET ${TARGET} APPEND PROPERTY BUILD_RPATH "$ORIGIN")
            set_property(TARGET ${TARGET} APPEND PROPERTY INSTALL_RPATH "$ORIGIN")
        endif()
    endif()
endfunction()
