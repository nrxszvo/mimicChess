add_library(pgnProcessing eloWriter.cpp parallelParser.cpp parseMoves.cpp decompress.cpp)
target_link_libraries(utils)
target_include_directories(pgnProcessing PUBLIC "${PROJECT_SOURCE_DIR}")
target_include_directories(pgnProcessing PUBLIC "${PROJECT_SOURCE_DIR}/include")
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	set(CMAKE_PREFIXPATH /opt/homebrew)
endif()

find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
target_link_libraries(pgnProcessing PUBLIC Arrow::arrow_shared parquet_shared curl)

find_package(zstd REQUIRED)
target_link_libraries(pgnProcessing PRIVATE zstd::libzstd)

find_package(re2 REQUIRED)
target_link_libraries(pgnProcessing PRIVATE re2::re2)
