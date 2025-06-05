# Build as a shared library for Python bindings
add_library(pgnProcessing SHARED
    eloWriter.cpp
    parallelParser.cpp
    parseMoves.cpp
    decompress.cpp
)

# Link against utils library
target_link_libraries(pgnProcessing PRIVATE utils)

# Include directories
target_include_directories(pgnProcessing PUBLIC
    "${PROJECT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/include"
)

# System-specific settings
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_PREFIXPATH /opt/homebrew)
    # Make sure RPATH is set correctly for macOS
    set_target_properties(pgnProcessing PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "@loader_path"
    )
endif()

# Find and link dependencies
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
find_package(zstd REQUIRED)
find_package(re2 REQUIRED)

# Link dependencies
target_link_libraries(pgnProcessing
    PUBLIC
        Arrow::arrow_shared
        parquet_shared
        curl
    PRIVATE
        zstd::libzstd
        re2::re2
)
