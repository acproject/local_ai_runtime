cmake_minimum_required(VERSION 3.20)

project(local_ai_runtime LANGUAGES CXX)

include(FetchContent)

option(LOCAL_AI_RUNTIME_WITH_LLAMA_CPP "Build with llama.cpp provider" ON)
option(LOCAL_AI_RUNTIME_LLAMA_CUDA "Enable CUDA backend for llama.cpp (NVIDIA)" OFF)
option(LOCAL_AI_RUNTIME_LLAMA_VULKAN "Enable Vulkan backend for llama.cpp" OFF)
set(LOCAL_AI_RUNTIME_CUDA_ARCHITECTURES "" CACHE STRING "CMake CUDA architectures for ggml-cuda (e.g. 86-real;89-real). Empty = safe default")

FetchContent_Declare(
  cpp_httplib
  URL https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.15.3/httplib.h
  DOWNLOAD_NO_EXTRACT TRUE
)
FetchContent_Populate(cpp_httplib)
add_library(cpp-httplib INTERFACE)
target_include_directories(cpp-httplib INTERFACE ${cpp_httplib_SOURCE_DIR})

FetchContent_Declare(
  nlohmann_json
  URL https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp
  DOWNLOAD_NO_EXTRACT TRUE
  DOWNLOAD_NAME json.hpp
)
FetchContent_Populate(nlohmann_json)
file(MAKE_DIRECTORY ${nlohmann_json_SOURCE_DIR}/nlohmann)
configure_file(
  ${nlohmann_json_SOURCE_DIR}/json.hpp
  ${nlohmann_json_SOURCE_DIR}/nlohmann/json.hpp
  COPYONLY
)
add_library(nlohmann-json INTERFACE)
target_include_directories(nlohmann-json INTERFACE ${nlohmann_json_SOURCE_DIR})

if(LOCAL_AI_RUNTIME_WITH_LLAMA_CPP)
  if(LOCAL_AI_RUNTIME_LLAMA_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    if(LOCAL_AI_RUNTIME_CUDA_ARCHITECTURES STREQUAL "")
      set(CMAKE_CUDA_ARCHITECTURES "75-virtual;80-virtual;86-real;89-real;120a-real" CACHE STRING "" FORCE)
    else()
      set(CMAKE_CUDA_ARCHITECTURES "${LOCAL_AI_RUNTIME_CUDA_ARCHITECTURES}" CACHE STRING "" FORCE)
    endif()
  endif()
  if(LOCAL_AI_RUNTIME_LLAMA_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
  endif()

  FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG b8003
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(llama_cpp)
endif()

add_executable(local-ai-runtime
  src/config.cpp
  src/main.cpp
  src/mcp_client.cpp
  src/openai_compatible_http_provider.cpp
  src/ollama_provider.cpp
  src/openai_router.cpp
  src/session_manager.cpp
  src/tooling.cpp
  extern/llama_cpp_agent/src/gbnf_generator.cpp
  extern/llama_cpp_agent/src/tool_call_parser.cpp
  extern/llama_cpp_agent/src/tool_manager.cpp
)

if(LOCAL_AI_RUNTIME_WITH_LLAMA_CPP)
  target_sources(local-ai-runtime PRIVATE
    src/llama_cpp_provider.cpp
  )
endif()

set_target_properties(local-ai-runtime PROPERTIES
  CXX_STANDARD 23
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
  PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
)

target_include_directories(local-ai-runtime PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/llama_cpp_agent/include
)
target_link_libraries(local-ai-runtime PRIVATE cpp-httplib nlohmann-json)

if(LOCAL_AI_RUNTIME_WITH_LLAMA_CPP)
  target_compile_definitions(local-ai-runtime PRIVATE LOCAL_AI_RUNTIME_WITH_LLAMA_CPP=1)
else()
  target_compile_definitions(local-ai-runtime PRIVATE LOCAL_AI_RUNTIME_WITH_LLAMA_CPP=0)
endif()

if(LOCAL_AI_RUNTIME_WITH_LLAMA_CPP)
  target_link_libraries(local-ai-runtime PRIVATE llama)
endif()

if(MSVC)
  target_compile_options(local-ai-runtime PRIVATE /W4)
else()
  target_compile_options(local-ai-runtime PRIVATE -Wall -Wextra -Wpedantic)
endif()
