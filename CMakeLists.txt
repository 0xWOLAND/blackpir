cmake_minimum_required(VERSION 3.14)
project(SimplePIR LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)

# Include directories
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Source files
set(SOURCES
    pir.cc
)

set(HEADERS
    pir.h
)

# Create library
add_library(simple_pir ${SOURCES} ${HEADERS})

# Enable testing
enable_testing()

# Test sources
set(TEST_SOURCES
    tests/pir_test.cc
)

# Create test executable
add_executable(pir_tests ${TEST_SOURCES})
target_link_libraries(pir_tests 
    PRIVATE 
        simple_pir
        GTest::gtest 
        GTest::gtest_main
)

# Add tests
add_test(NAME PIRTest COMMAND pir_tests)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(simple_pir PRIVATE
        -Wall -Wextra -Wpedantic
        -O2
    )
    target_compile_options(pir_tests PRIVATE
        -Wall -Wextra -Wpedantic
        -O2
    )
elseif(MSVC)
    target_compile_options(simple_pir PRIVATE
        /W4
        /O2
    )
    target_compile_options(pir_tests PRIVATE
        /W4
        /O2
    )
endif()

# Install rules
install(TARGETS simple_pir
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES ${HEADERS} DESTINATION include)