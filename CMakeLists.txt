cmake_minimum_required(VERSION 3.20)

project(NeuroSwap LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(neuroswap_core
    src/controller/controller.cpp
    src/pinner/memory_pinner.cpp
    src/prefetcher/file_mapping.cpp
    src/profiler/measurement.cpp
    src/workload/memory_workload.cpp
)

target_include_directories(neuroswap_core PUBLIC include)

target_compile_options(neuroswap_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(neuroswap_baseline src/main.cpp)
target_link_libraries(neuroswap_baseline PRIVATE neuroswap_core)

add_executable(neuroswap_pinned src/pinned_main.cpp)
target_link_libraries(neuroswap_pinned PRIVATE neuroswap_core)

add_executable(neuroswap_prefetch src/prefetch_main.cpp)
target_link_libraries(neuroswap_prefetch PRIVATE neuroswap_core)

add_executable(neuroswap_controller src/controller_main.cpp)
target_link_libraries(neuroswap_controller PRIVATE neuroswap_core)

include(CTest)
if(BUILD_TESTING)
    add_executable(memory_workload_test tests/memory_workload_test.cpp)
    target_link_libraries(memory_workload_test PRIVATE neuroswap_core)
    add_test(NAME memory_workload_test COMMAND memory_workload_test)
    add_executable(memory_pinner_test tests/memory_pinner_test.cpp)
    target_link_libraries(memory_pinner_test PRIVATE neuroswap_core)
    add_test(NAME memory_pinner_test COMMAND memory_pinner_test)
    add_executable(file_mapping_test tests/file_mapping_test.cpp)
    target_link_libraries(file_mapping_test PRIVATE neuroswap_core)
    add_test(NAME file_mapping_test COMMAND file_mapping_test)
    add_executable(controller_test tests/controller_test.cpp)
    target_link_libraries(controller_test PRIVATE neuroswap_core)
    add_test(NAME controller_test COMMAND controller_test)
endif()
