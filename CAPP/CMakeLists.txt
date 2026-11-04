cmake_minimum_required(VERSION 3.10)
project(Undistortion)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ===================================================================
# Compiler Specific Configuration
# ===================================================================

if(MSVC)
    # MSVC Compiler Settings
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi /Od /MDd")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi /O2 /MD")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /Zi /O2 /MD")

    # Enable AVX2 for MSVC
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /arch:AVX2")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /arch:AVX2")

    # Enable FMA for MSVC
    add_compile_definitions(__FMA__)

    # MSVC specific options
    add_compile_options(/MP)          # Multi-processor compilation
    add_compile_options(/utf-8)       # Use UTF-8 source files
    add_compile_options(/wd4828)      # Disable warning 4828
    add_compile_options(/wd4819)      # Disable warning 4819

    # Definitions
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-DUNICODE -D_UNICODE)
    add_definitions(-D_WIN32_WINNT=0x0601)

    # Remove debug flags for MSVC
    string(REPLACE "-g" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REPLACE "-g" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "-g" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
else()
    # GCC/Clang settings
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mavx2 -mfma")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O2 -g -march=native -mavx2 -mfma")

    # GCC/Clang specific options
    add_compile_options(-Wall -Wextra)

    # Enable FMA definition for GCC/Clang
    add_compile_definitions(__FMA__)
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
)

# Find OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found!")
endif()
include_directories(${OpenCV_INCLUDE_DIRS})

# Collect source files
file(GLOB_RECURSE UNDISTORTION_HEADERS ${CMAKE_SOURCE_DIR}/include/*.h)
file(GLOB_RECURSE UNDISTORTION_SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp)

# ===================================================================
# Target: Undistortion (Shared Library)
# ===================================================================
add_library(Undistortion SHARED
    ${UNDISTORTION_SOURCES}
    ${UNDISTORTION_HEADERS}
)

# Link OpenCV
target_link_libraries(Undistortion PRIVATE ${OpenCV_LIBS})

# Add compile definitions
target_compile_definitions(Undistortion PRIVATE
    UNDISTORTION_ENABLE_LOGGING
    AOI_EXPORTS  # Required for __declspec(dllexport) in C API
)

# Set Debug suffix for DLL name
set_target_properties(Undistortion PROPERTIES
    DEBUG_POSTFIX "_d"
)

# Set output directories for Windows
if(WIN32)
    set_target_properties(Undistortion PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin"
        ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/lib"
        ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/lib"
    )
endif()

# Additional output directory configuration for all platforms
set_target_properties(Undistortion PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# ===================================================================
# Target: test_undistortion (Executable)
# ===================================================================
# Filter out C API files for the C++ test executable
set(TEST_SOURCES ${UNDISTORTION_SOURCES})
list(FILTER TEST_SOURCES EXCLUDE REGEX ".*_C_API\\.cpp$")

add_executable(test_undistortion
    test/test_undistortion.cpp
    ${TEST_SOURCES}
    ${UNDISTORTION_HEADERS}
)

target_link_libraries(test_undistortion PRIVATE ${OpenCV_LIBS})

target_compile_definitions(test_undistortion PRIVATE
    UNDISTORTION_ENABLE_LOGGING
)

if(WIN32)
    set_target_properties(test_undistortion PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# ===================================================================
# Target: bench_undistortion (Release benchmark + bit-exact self check)
# ===================================================================
set(BENCH_SOURCES ${UNDISTORTION_SOURCES})
list(FILTER BENCH_SOURCES EXCLUDE REGEX ".*_C_API\\.cpp$")

add_executable(bench_undistortion
    test/bench_undistortion.cpp
    ${BENCH_SOURCES}
    ${UNDISTORTION_HEADERS}
)

target_link_libraries(bench_undistortion PRIVATE ${OpenCV_LIBS})

target_compile_definitions(bench_undistortion PRIVATE
    UNDISTORTION_ENABLE_LOGGING
)

if(WIN32)
    set_target_properties(bench_undistortion PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# Print configuration info
message(STATUS "")
message(STATUS "=== Undistortion Project Configuration ===")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "Architecture: AVX2/FMA enabled")
message(STATUS "Targets: Undistortion (Shared Lib), test_undistortion (Exe)")
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Debug DLL: Undistortion_d.dll")
else()
    message(STATUS "Release DLL: Undistortion.dll")
endif()
message(STATUS "")

# Installation rules
install(TARGETS Undistortion test_undistortion
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)
