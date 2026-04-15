# CMakeLists.txt - Neural Memory Adapter
# Production: cmake .. -DCMAKE_BUILD_TYPE=Release
# Sandbox:    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=x86-64"
cmake_minimum_required(VERSION 3.20)
project(neural-memory-adapter VERSION 1.0.0 LANGUAGES CXX)

option(USE_MSSQL "Build with MSSQL/ODBC support (requires unixodbc-dev)" ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(OPT_FLAGS -O3 -DNDEBUG -march=x86-64 -flto)
set(WARN_FLAGS -Wall -Wextra -Wpedantic -Wno-unused-parameter)

include(CheckCXXCompilerFlag)
set(HAS_AVX2 FALSE)
# AVX-512: disabled by default (Ryzen 7 3800X doesn't have it)
set(HAS_AVX512 FALSE)

# sandbox

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    add_definitions(-DHAS_OPENMP=1)
endif()

# MSSQL / ODBC (optional)
if(USE_MSSQL)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(ODBC QUIET odbc)
    endif()
    if(NOT ODBC_FOUND)
        find_path(ODBC_INCLUDE_DIR sql.h PATHS /usr/include /usr/local/include PATH_SUFFIXES odbc)
        find_library(ODBC_LIBRARY NAMES odbc unixodbc)
        if(ODBC_INCLUDE_DIR AND ODBC_LIBRARY)
            set(ODBC_FOUND TRUE)
            set(ODBC_INCLUDE_DIRS ${ODBC_INCLUDE_DIR})
            set(ODBC_LIBRARIES ${ODBC_LIBRARY})
        endif()
    endif()
    if(ODBC_FOUND)
        message(STATUS "MSSQL: ODBC found, building with MSSQL support")
        add_definitions(-DUSE_MSSQL=1)
    else()
        message(WARNING "MSSQL: ODBC not found, disabling MSSQL support")
        set(USE_MSSQL OFF)
    endif()
else()
    message(STATUS "MSSQL: disabled (-DUSE_MSSQL=OFF)")
endif()

include_directories(${CMAKE_SOURCE_DIR}/include ${ODBC_INCLUDE_DIRS})

set(NEURAL_SOURCES
    src/simd/simd_engine.cpp
    src/memory/hopfield.cpp src/memory/consolidation.cpp
    src/memory/memory_manager.cpp src/memory/vsa.cpp
    src/memory/knn.cpp
    src/memory/lstm.cpp
    src/graph/knowledge_graph.cpp
    src/core/memory_adapter.cpp
    src/core/c_api.cpp
)

if(USE_MSSQL AND ODBC_FOUND)
    list(APPEND NEURAL_SOURCES
        src/mssql/connection_pool.cpp
        src/mssql/bulk_ops.cpp
        src/mssql/vector_store.cpp
    )
endif()

add_library(neural_memory SHARED ${NEURAL_SOURCES})
target_compile_options(neural_memory PRIVATE ${OPT_FLAGS} ${WARN_FLAGS})
target_link_libraries(neural_memory PRIVATE
    $<$<BOOL:${USE_MSSQL}>:${ODBC_LIBRARIES}>
    $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
    pthread stdc++
)
set_target_properties(neural_memory PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 1)

add_executable(bench_vector_ops src/benchmarks/bench_vector_ops.cpp)
target_compile_options(bench_vector_ops PRIVATE ${OPT_FLAGS})
target_link_libraries(bench_vector_ops PRIVATE neural_memory)

add_executable(bench_memory src/benchmarks/bench_memory.cpp)
target_compile_options(bench_memory PRIVATE ${OPT_FLAGS})
target_link_libraries(bench_memory PRIVATE neural_memory)

add_executable(neural_memory_demo src/main.cpp)
target_compile_options(neural_memory_demo PRIVATE ${OPT_FLAGS} ${WARN_FLAGS})
target_link_libraries(neural_memory_demo PRIVATE neural_memory)

enable_testing()
foreach(t test_vector_ops test_hopfield test_graph)
    add_executable(${t} tests/${t}.cpp)
    target_compile_options(${t} PRIVATE ${OPT_FLAGS} -g)
    target_link_libraries(${t} PRIVATE neural_memory)
    add_test(NAME ${t} COMMAND ${t})
endforeach()

install(TARGETS neural_memory LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(DIRECTORY include/neural DESTINATION include)
install(TARGETS neural_memory_demo bench_vector_ops bench_memory RUNTIME DESTINATION bin)
