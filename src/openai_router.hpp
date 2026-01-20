#pragma once

#include "providers/registry.hpp"
#include "session_manager.hpp"
#include "tooling.hpp"

#include <httplib.h>

#include <memory>

namespace runtime {

class OpenAiRouter {
 public:
  OpenAiRouter(SessionManager* sessions, ProviderRegistry* providers, ToolRegistry tools);
  void Register(httplib::Server* server);
  ToolRegistry* MutableTools();

 private:
  SessionManager* sessions_;
  ProviderRegistry* providers_;
  ToolRegistry tools_;
};

}  // namespace runtime
