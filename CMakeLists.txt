cmake_minimum_required(VERSION 3.20)

project(local_ai_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

option(LOCAL_AI_RUNTIME_WITH_LLAMA_CPP "Build with llama.cpp provider" ON)

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
  FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
    GIT_TAG b4126
  )
  FetchContent_MakeAvailable(llama_cpp)
endif()

add_executable(local-ai-runtime
  src/config.cpp
  src/main.cpp
  src/mcp_client.cpp
  src/llama_cpp_provider.cpp
  src/openai_compatible_http_provider.cpp
  src/ollama_provider.cpp
  src/openai_router.cpp
  src/session_manager.cpp
  src/tooling.cpp
)

target_include_directories(local-ai-runtime PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(local-ai-runtime PRIVATE cpp-httplib nlohmann-json)

if(LOCAL_AI_RUNTIME_WITH_LLAMA_CPP)
  target_link_libraries(local-ai-runtime PRIVATE llama)
endif()

if(MSVC)
  target_compile_options(local-ai-runtime PRIVATE /W4)
else()
  target_compile_options(local-ai-runtime PRIVATE -Wall -Wextra -Wpedantic)
endif()
