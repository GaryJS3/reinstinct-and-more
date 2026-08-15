cmake_minimum_required(VERSION 3.21)
project(reinstinct_mtmd LANGUAGES CXX)

# The bridge is shared while the pinned Furnace libraries are embedded static
# archives, so every transitive object must be position-independent.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# llama.cpp normally initializes this when configured as the top-level
# project. Supplying it here keeps its mtmd target valid as a subdirectory.
set(LLAMA_INSTALL_VERSION "0.0.0" CACHE STRING "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_HIP ON CACHE BOOL "Build libmtmd vision encoder with HIP" FORCE)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/llama.cpp llama.cpp)
# mtmd lives below llama.cpp's tools tree. Add only that library rather than
# enabling every llama.cpp command-line tool.
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/llama.cpp/tools/mtmd mtmd)

add_library(reinstinct_mtmd SHARED reinstinct_mtmd.cpp reinstinct_mtmd.h)
target_compile_features(reinstinct_mtmd PRIVATE cxx_std_17)
target_include_directories(reinstinct_mtmd PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/llama.cpp/include
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/llama.cpp/ggml/include
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/llama.cpp/tools/mtmd)
target_link_libraries(reinstinct_mtmd PRIVATE mtmd llama)
set_target_properties(reinstinct_mtmd PROPERTIES OUTPUT_NAME reinstinct_mtmd)
