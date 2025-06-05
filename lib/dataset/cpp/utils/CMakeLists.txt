find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)

# Build as a shared library
add_library(utils SHARED
    utils.cpp
    parquetReader.cpp
    parquetWriter.cpp
)

# Link dependencies
target_link_libraries(utils
    PUBLIC
        Arrow::arrow_shared
        parquet_shared
        curl
)

# Include directories
target_include_directories(utils
    PUBLIC
        "${PROJECT_SOURCE_DIR}"
)

# Set RPATH for macOS
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set_target_properties(utils PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "@loader_path"
    )
endif()