# Findc2pa.cmake — locate the c2pa-c library (C bindings for the Rust c2pa crate).
#
# Pre-built binaries: https://github.com/contentauth/c2pa-c/releases
# Install: copy libc2pa.so (or .dylib / .dll) + c2pa.h to a prefix.
#
# Sets:
#   c2pa_FOUND       — TRUE if header + library found
#   c2pa_INCLUDE_DIR — path to c2pa.h
#   c2pa_LIBRARY     — path to libc2pa
#
# Usage in CMakeLists.txt:
#   find_package(c2pa)
#   if(c2pa_FOUND)
#       target_compile_definitions(my_target PRIVATE CRISPASR_HAVE_C2PA)
#       target_include_directories(my_target PRIVATE ${c2pa_INCLUDE_DIR})
#       target_link_libraries(my_target PRIVATE ${c2pa_LIBRARY})
#   endif()

find_path(c2pa_INCLUDE_DIR
    NAMES c2pa.h c2pa/c2pa.h
    HINTS ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES include
)

find_library(c2pa_LIBRARY
    NAMES c2pa libc2pa
    HINTS ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(c2pa
    REQUIRED_VARS c2pa_LIBRARY c2pa_INCLUDE_DIR
)

mark_as_advanced(c2pa_INCLUDE_DIR c2pa_LIBRARY)
