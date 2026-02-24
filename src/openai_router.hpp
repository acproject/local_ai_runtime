#pragma once

#include "providers/registry.hpp"
#include "session_manager.hpp"

#include <httplib.h>

#include <memory>

namespace runtime {

class OpenAiRouter {
 public:
  OpenAiRouter(SessionManager* sessions, ProviderRegistry* providers);
  void Register(httplib::Server* server);

 private:
  SessionManager* sessions_;
  ProviderRegistry* providers_;
};

}  // namespace runtime
