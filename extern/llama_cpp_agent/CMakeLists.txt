cmake_minimum_required(VERSION 3.20)
project(llama_cpp_agent VERSION 0.1.0 LANGUAGES CXX)

# C++23 Standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(LLAMA_CPP_ENABLE "Enable llama.cpp integration" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Werror=return-type
        -Werror=uninitialized
    )
elseif(MSVC)
    add_compile_options(
        /W4
        /WX-
        /permissive-
    )
endif()

# Find threads
find_package(Threads REQUIRED)

# Add llama.cpp
if(LLAMA_CPP_ENABLE)
    set(LLAMA_STANDALONE ON)
    set(BUILD_SHARED_LIBS OFF)
    add_subdirectory(extern/llama.cpp)
    set(LLAMA_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/extern/llama.cpp/include)
    set(LLAMA_COMMON_DIR ${CMAKE_SOURCE_DIR}/extern/llama.cpp/common)
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/extern/cpp-httplib
    ${CMAKE_SOURCE_DIR}/extern/json/include
    ${CMAKE_SOURCE_DIR}/extern/llama.cpp/vendor
    ${LLAMA_INCLUDE_DIR}
    ${LLAMA_COMMON_DIR}
)

# Main library
add_library(llama_agent_lib STATIC
    src/agent_runtime.cpp
    src/conversation.cpp
    src/gbnf_generator.cpp
    src/llama_wrapper.cpp
    src/tool_manager.cpp
    src/tool_call_parser.cpp
    src/http_server.cpp
)

target_link_libraries(llama_agent_lib
    PUBLIC
        llama
        Threads::Threads
)

target_include_directories(llama_agent_lib
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/extern/cpp-httplib
        ${CMAKE_SOURCE_DIR}/extern/json/include
)

# Main executable
add_executable(llama_agent_server
    src/main.cpp
)

target_link_libraries(llama_agent_server
    PRIVATE
        llama_agent_lib
)

# Tests
if(BUILD_TESTS)
    enable_testing()
    include(FetchContent)
    
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
    
    add_executable(test_gbnf_generator
        tests/test_gbnf_generator.cpp
        src/gbnf_generator.cpp
    )
    
    target_link_libraries(test_gbnf_generator
        PRIVATE
            GTest::gtest_main
            llama_agent_lib
    )
    
    add_test(NAME test_gbnf_generator COMMAND test_gbnf_generator)
endif()

# Install
install(TARGETS llama_agent_server
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/llama_agent
    DESTINATION include
)

# Print configuration
message(STATUS "")
message(STATUS "Project Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  llama.cpp: ${LLAMA_CPP_ENABLE}")
message(STATUS "  Tests: ${BUILD_TESTS}")
message(STATUS "")
