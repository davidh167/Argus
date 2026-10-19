cmake_minimum_required(VERSION 3.24)
project(argus_core LANGUAGES C CXX)  # usearch (fetched below) enables C features

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(ARGUS_SANITIZE "Build with AddressSanitizer + UBSan" OFF)
option(ARGUS_BUILD_TESTS "Build the Catch2 tests and Google Benchmark binaries" ON)

# Must come before any target is defined so the library itself is instrumented, not just the tests.
if(ARGUS_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

find_package(SQLite3 REQUIRED)
find_package(llama CONFIG REQUIRED)  # Homebrew: brew install llama.cpp

# usearch: header-only HNSW vector index. Pinned release, fetched from GitHub at configure time.
include(FetchContent)
set(USEARCH_BUILD_TEST_CPP OFF CACHE BOOL "" FORCE)
set(USEARCH_BUILD_BENCH_CPP OFF CACHE BOOL "" FORCE)
set(USEARCH_BUILD_LIB_C OFF CACHE BOOL "" FORCE)
set(USEARCH_BUILD_SQLITE OFF CACHE BOOL "" FORCE)
set(USEARCH_USE_OPENMP OFF CACHE BOOL "" FORCE)
FetchContent_Declare(usearch GIT_REPOSITORY https://github.com/unum-cloud/usearch.git GIT_TAG v2.26.5 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(usearch)

add_library(argus SHARED src/argus.cpp src/embedder.cpp)
target_include_directories(argus PUBLIC include)
target_link_libraries(argus PRIVATE SQLite::SQLite3 llama usearch)
target_include_directories(argus SYSTEM PRIVATE ${usearch_SOURCE_DIR}/include)
target_compile_options(argus PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(argus PROPERTIES CXX_VISIBILITY_PRESET hidden)

# Static build for linking straight into the Rust app (no dylib to ship or locate).
add_library(argus_static STATIC src/argus.cpp src/embedder.cpp)
target_include_directories(argus_static PUBLIC include)
target_link_libraries(argus_static PUBLIC SQLite::SQLite3 llama usearch)
target_include_directories(argus_static SYSTEM PRIVATE ${usearch_SOURCE_DIR}/include)
target_compile_options(argus_static PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(argus_static PROPERTIES CXX_VISIBILITY_PRESET hidden POSITION_INDEPENDENT_CODE ON)

install(TARGETS argus_static ARCHIVE DESTINATION lib)
install(FILES include/argus.h DESTINATION include)

if(ARGUS_BUILD_TESTS)
enable_testing()
find_package(Catch2 3 REQUIRED)
add_executable(argus_tests tests/test_index.cpp tests/test_embedder.cpp tests/test_vectors.cpp)
target_link_libraries(argus_tests PRIVATE argus Catch2::Catch2WithMain)
include(Catch)
# Catch2 from Homebrew is not built with ASan; libc++ container-overflow annotations then disagree
# between it and our instrumented code and report false positives inside Catch2. Disable only
# that one check; heap overflows, use-after-free, leaks and UBSan stay on.
catch_discover_tests(argus_tests PROPERTIES ENVIRONMENT "ASAN_OPTIONS=detect_container_overflow=0")

find_package(benchmark REQUIRED)
add_executable(argus_bench bench/bench_search.cpp)
target_link_libraries(argus_bench PRIVATE argus benchmark::benchmark)
endif()
