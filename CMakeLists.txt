cmake_minimum_required(VERSION 3.20)

project(local_ai_runtime LANGUAGES CXX)

include(FetchContent)

option(LOCAL_AI_RUNTIME_WITH_LLAMA_CPP "Build with llama.cpp provider" ON)
option(LOCAL_AI_RUNTIME_LLAMA_CUDA "Enable CUDA backend for llama.cpp (NVIDIA)" OFF)
option(LOCAL_AI_RUNTIME_LLAMA_VULKAN "Enable Vulkan backend for llama.cpp" OFF)
option(LOCAL_AI_RUNTIME_USE_LLAMA_CPP_AGENT "Build llama.cpp-agent tools (optional)" OFF)
option(LOCAL_AI_RUNTIME_FETCH_LLAMA_CPP_AGENT "Fetch llama.cpp-agent when submodule is missing" OFF)
option(LOCAL_AI_RUNTIME_FETCH_LLAMA_CPP "Fetch llama.cpp when it is missing" ON)
set(LOCAL_AI_RUNTIME_LLAMA_CPP_SOURCE_DIR "" CACHE PATH "Path to a local llama.cpp checkout (optional)")
set(LOCAL_AI_RUNTIME_CUDA_ARCHITECTURES "" CACHE STRING "CMake CUDA architectures for ggml-cuda (e.g. 86-real;89-real). Empty = safe default")

set(LOCAL_AI_RUNTIME_VENDORED_HTTPLIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp-agent/extern/cpp-httplib")
set(LOCAL_AI_RUNTIME_CACHED_HTTPLIB_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}/build-trae/_deps/cpp_httplib-src"
  "${CMAKE_CURRENT_SOURCE_DIR}/build-vs2022-x64-cuda/_deps/cpp_httplib-src"
)
set(LOCAL_AI_RUNTIME_HTTPLIB_INCLUDE_DIR "")
if(EXISTS "${LOCAL_AI_RUNTIME_VENDORED_HTTPLIB_DIR}/httplib.h")
  set(LOCAL_AI_RUNTIME_HTTPLIB_INCLUDE_DIR "${LOCAL_AI_RUNTIME_VENDORED_HTTPLIB_DIR}")
else()
  foreach(dir IN LISTS LOCAL_AI_RUNTIME_CACHED_HTTPLIB_DIRS)
    if(EXISTS "${dir}/httplib.h")
      set(LOCAL_AI_RUNTIME_HTTPLIB_INCLUDE_DIR "${dir}")
      break()
    endif()
  endforeach()
endif()
if(NOT LOCAL_AI_RUNTIME_HTTPLIB_INCLUDE_DIR STREQUAL "")
  add_library(local-ai-cpp-httplib INTERFACE)
  target_include_directories(local-ai-cpp-httplib INTERFACE "${LOCAL_AI_RUNTIME_HTTPLIB_INCLUDE_DIR}")
else()
  FetchContent_Declare(
    cpp_httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.15.3
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(cpp_httplib)
  add_library(local-ai-cpp-httplib INTERFACE)
  target_include_directories(local-ai-cpp-httplib INTERFACE ${cpp_httplib_SOURCE_DIR})
endif()

set(LOCAL_AI_RUNTIME_VENDORED_NLOHMANN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp-agent/extern/json/include")
set(LOCAL_AI_RUNTIME_CACHED_NLOHMANN_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}/build-vs2022-x64-cuda/_deps/nlohmann_json-src"
  "${CMAKE_CURRENT_SOURCE_DIR}/build-vs2022-x64-cuda/_deps/llama_cpp-src/vendor"
)
set(LOCAL_AI_RUNTIME_NLOHMANN_INCLUDE_DIR "")
if(EXISTS "${LOCAL_AI_RUNTIME_VENDORED_NLOHMANN_DIR}/nlohmann/json.hpp")
  set(LOCAL_AI_RUNTIME_NLOHMANN_INCLUDE_DIR "${LOCAL_AI_RUNTIME_VENDORED_NLOHMANN_DIR}")
else()
  foreach(dir IN LISTS LOCAL_AI_RUNTIME_CACHED_NLOHMANN_DIRS)
    if(EXISTS "${dir}/nlohmann/json.hpp")
      set(LOCAL_AI_RUNTIME_NLOHMANN_INCLUDE_DIR "${dir}")
      break()
    endif()
  endforeach()
endif()
if(NOT LOCAL_AI_RUNTIME_NLOHMANN_INCLUDE_DIR STREQUAL "")
  add_library(nlohmann-json INTERFACE)
  target_include_directories(nlohmann-json INTERFACE "${LOCAL_AI_RUNTIME_NLOHMANN_INCLUDE_DIR}")
else()
  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(nlohmann_json)
  add_library(nlohmann-json INTERFACE)
  target_include_directories(nlohmann-json INTERFACE
    ${nlohmann_json_SOURCE_DIR}
    ${nlohmann_json_SOURCE_DIR}/single_include
  )
endif()

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

  set(LOCAL_AI_RUNTIME_LLAMA_CPP_SOURCE_CANDIDATES
    "${LOCAL_AI_RUNTIME_LLAMA_CPP_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/build/_deps/llama_cpp-src"
    "${CMAKE_CURRENT_SOURCE_DIR}/build-vs2022-x64-cuda/_deps/llama_cpp-src"
    "${CMAKE_CURRENT_SOURCE_DIR}/build-trae/_deps/llama_cpp-src"
  )

  set(LOCAL_AI_RUNTIME_LLAMA_CPP_FOUND_SOURCE_DIR "")
  foreach(dir IN LISTS LOCAL_AI_RUNTIME_LLAMA_CPP_SOURCE_CANDIDATES)
    if(NOT dir STREQUAL "" AND EXISTS "${dir}/CMakeLists.txt")
      set(LOCAL_AI_RUNTIME_LLAMA_CPP_FOUND_SOURCE_DIR "${dir}")
      break()
    endif()
  endforeach()

  if(LOCAL_AI_RUNTIME_LLAMA_CPP_FOUND_SOURCE_DIR STREQUAL "" AND LOCAL_AI_RUNTIME_FETCH_LLAMA_CPP)
    FetchContent_Declare(
      llama_cpp
      GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
      GIT_TAG master
      GIT_SHALLOW TRUE
    )
    FetchContent_GetProperties(llama_cpp)
    if(NOT llama_cpp_POPULATED)
      if(POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
      endif()
      FetchContent_Populate(llama_cpp)
    endif()
    set(LOCAL_AI_RUNTIME_LLAMA_CPP_FOUND_SOURCE_DIR "${llama_cpp_SOURCE_DIR}")
  endif()

  if(LOCAL_AI_RUNTIME_LLAMA_CPP_FOUND_SOURCE_DIR STREQUAL "")
    message(FATAL_ERROR "llama.cpp source not found. Provide LOCAL_AI_RUNTIME_LLAMA_CPP_SOURCE_DIR, initialize third_party/llama.cpp-agent submodules, or enable LOCAL_AI_RUNTIME_FETCH_LLAMA_CPP.")
  endif()

  # Enable building common library for jinja chat template support
  set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)

  add_subdirectory("${LOCAL_AI_RUNTIME_LLAMA_CPP_FOUND_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/llama_cpp")

  if(LOCAL_AI_RUNTIME_USE_LLAMA_CPP_AGENT)
    set(LLAMA_CPP_SOURCE_DIR_OVERRIDE "${LOCAL_AI_RUNTIME_LLAMA_CPP_FOUND_SOURCE_DIR}" CACHE PATH "" FORCE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp-agent/CMakeLists.txt")
      add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp-agent" "${CMAKE_BINARY_DIR}/llama_cpp_agent")
    elseif(LOCAL_AI_RUNTIME_FETCH_LLAMA_CPP_AGENT)
      FetchContent_Declare(
        llama_cpp_agent
        GIT_REPOSITORY https://github.com/acproject/llama.cpp-agent.git
        GIT_TAG main
        GIT_SHALLOW TRUE
      )
      FetchContent_MakeAvailable(llama_cpp_agent)
    endif()
  endif()
endif()

add_executable(local-ai-runtime
  src/config.cpp
  src/main.cpp
  src/openai_compatible_http_provider.cpp
  src/ollama_provider.cpp
  src/openai_router.cpp
  src/session_manager.cpp
  src/agent_server_provider.cpp
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
)
target_link_libraries(local-ai-runtime PRIVATE local-ai-cpp-httplib nlohmann-json)

if(LOCAL_AI_RUNTIME_WITH_LLAMA_CPP)
  target_compile_definitions(local-ai-runtime PRIVATE LOCAL_AI_RUNTIME_WITH_LLAMA_CPP=1)
  target_include_directories(local-ai-runtime PRIVATE
    "${LOCAL_AI_RUNTIME_LLAMA_CPP_FOUND_SOURCE_DIR}/common"
  )
else()
  target_compile_definitions(local-ai-runtime PRIVATE LOCAL_AI_RUNTIME_WITH_LLAMA_CPP=0)
endif()

if(LOCAL_AI_RUNTIME_WITH_LLAMA_CPP)
  target_link_libraries(local-ai-runtime PRIVATE llama common)
endif()

if(MSVC)
  target_compile_options(local-ai-runtime PRIVATE /W4)
else()
  target_compile_options(local-ai-runtime PRIVATE -Wall -Wextra -Wpedantic)
endif()
