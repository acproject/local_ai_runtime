#pragma once

#include "ollama_provider.hpp"
#include "session_manager.hpp"
#include "tooling.hpp"

#include <httplib.h>

#include <memory>

namespace runtime {

class OpenAiRouter {
 public:
  OpenAiRouter(SessionManager* sessions, OllamaProvider* ollama, ToolRegistry tools);
  void Register(httplib::Server* server);
  ToolRegistry* MutableTools();

 private:
  SessionManager* sessions_;
  OllamaProvider* ollama_;
  ToolRegistry tools_;
};

}  // namespace runtime
