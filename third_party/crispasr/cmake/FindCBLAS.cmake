# FindCBLAS — locate a standalone C BLAS library (libcblas).
#
# Reference/Netlib BLAS (e.g. Debian/Ubuntu `libblas-dev` without OpenBLAS)
# ships the C interface symbols (cblas_sgemm, …) in a *separate* `libcblas`,
# while `find_package(BLAS)` only resolves the Fortran `libblas`. Linking the
# cblas symbols against libblas alone fails with:
#
#     undefined reference to `cblas_sgemm'
#
# Stock CMake provides FindBLAS but no FindCBLAS, so without this module a
# `find_package(CBLAS)` call silently finds nothing and the fix degrades to a
# no-op. This module does a direct library search so callers can prefer a real
# libcblas and fall back to find_package(BLAS) for OpenBLAS/MKL/Accelerate,
# which bundle the cblas symbols in their main library.
#
# Header discovery (cblas.h) is handled separately by callers via
# `find_path(... cblas.h PATH_SUFFIXES openblas)`.
#
# Sets: CBLAS_FOUND, CBLAS_LIBRARIES

find_library(CBLAS_LIBRARIES NAMES cblas)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CBLAS REQUIRED_VARS CBLAS_LIBRARIES)

mark_as_advanced(CBLAS_LIBRARIES)
