# Set the default compile features and properties for a target.

if (NOT TARGET)
    message(FATAL_ERROR "TARGET not set before including DefaultTargetOptions")
endif()

set(CRISPASR_CXX_STANDARD cxx_std_11)
if (GGML_CUDA AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
    set(CRISPASR_CXX_STANDARD cxx_std_17)
endif()

target_compile_features(${TARGET}
    PRIVATE
        ${CRISPASR_CXX_STANDARD}
    )

set_target_properties(${TARGET}
    PROPERTIES
        EXPORT_COMPILE_COMMANDS ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
