#include "openai_router.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <optional>
#include <string>
#include <unordered_set>

namespace runtime {
namespace {

static int64_t NowSeconds() {
  return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

static nlohmann::json MakeError(const std::string& message, const std::string& type) {
  nlohmann::json j;
  j["error"] = {{"message", message}, {"type", type}, {"param", nullptr}, {"code", nullptr}};
  return j;
}

static void SendJson(httplib::Response* res, int status, const nlohmann::json& body) {
  res->status = status;
  res->set_header("Content-Type", "application/json");
  res->set_content(body.dump(), "application/json");
}

static std::string SseData(const nlohmann::json& j) {
  return std::string("data: ") + j.dump() + "\n\n";
}

static std::string SseDone() {
  return "data: [DONE]\n\n";
}

static nlohmann::json ParseJsonBody(const httplib::Request& req) {
  return nlohmann::json::parse(req.body, nullptr, false);
}

static std::vector<ChatMessage> ParseChatMessages(const nlohmann::json& j, bool* ok) {
  std::vector<ChatMessage> out;
  *ok = false;
  if (!j.contains("messages") || !j["messages"].is_array()) return out;
  for (const auto& m : j["messages"]) {
    if (!m.is_object()) continue;
    ChatMessage cm;
    if (m.contains("role") && m["role"].is_string()) cm.role = m["role"].get<std::string>();
    if (m.contains("content") && m["content"].is_string()) cm.content = m["content"].get<std::string>();
    if (!cm.role.empty()) out.push_back(std::move(cm));
  }
  *ok = true;
  return out;
}

static std::vector<std::string> ParseRequestedToolNames(const nlohmann::json& j) {
  std::vector<std::string> out;
  if (!j.contains("tools") || !j["tools"].is_array()) return out;
  for (const auto& t : j["tools"]) {
    if (!t.is_object()) continue;
    if (t.contains("function") && t["function"].is_object() && t["function"].contains("name") &&
        t["function"]["name"].is_string()) {
      out.push_back(t["function"]["name"].get<std::string>());
    } else if (t.contains("name") && t["name"].is_string()) {
      out.push_back(t["name"].get<std::string>());
    }
  }
  return out;
}

static bool ToolChoiceIsNone(const nlohmann::json& j) {
  if (!j.contains("tool_choice")) return false;
  const auto& tc = j["tool_choice"];
  if (tc.is_string()) return tc.get<std::string>() == "none";
  if (tc.is_object() && tc.contains("type") && tc["type"].is_string()) return tc["type"].get<std::string>() == "none";
  return false;
}

static std::string BuildToolSystemPrompt(const std::vector<ToolSchema>& tools) {
  nlohmann::json tool_list = nlohmann::json::array();
  for (const auto& t : tools) {
    tool_list.push_back({{"name", t.name}, {"description", t.description}, {"parameters", t.parameters}});
  }
  nlohmann::json spec;
  spec["tools"] = tool_list;

  std::string prompt;
  prompt += "You are a tool-using assistant.\n";
  prompt += "If you need to call tools, respond ONLY with a single JSON object:\n";
  prompt += "{\"tool_calls\":[{\"id\":\"call_1\",\"name\":\"tool_name\",\"arguments\":{...}}]}\n";
  prompt += "If you can answer without tools, respond ONLY with:\n";
  prompt += "{\"final\":\"...\"}\n";
  prompt += "Never include any extra text outside the JSON.\n";
  prompt += "Available tools spec:\n";
  prompt += spec.dump();
  return prompt;
}

static std::string BuildPlannerSystemPrompt(const std::vector<ToolSchema>& tools, int max_plan_steps) {
  nlohmann::json tool_list = nlohmann::json::array();
  for (const auto& t : tools) {
    tool_list.push_back({{"name", t.name}, {"description", t.description}, {"parameters", t.parameters}});
  }
  nlohmann::json spec;
  spec["tools"] = tool_list;

  std::string prompt;
  prompt += "You are a planner.\n";
  prompt += "Return ONLY a single JSON object and no extra text.\n";
  prompt += "If tools are needed, output:\n";
  prompt += "{\"plan\":[{\"name\":\"tool_name\",\"arguments\":{...}}]}\n";
  prompt += "The plan length MUST be <= " + std::to_string(max_plan_steps) + ".\n";
  prompt += "If no tools are needed, output:\n";
  prompt += "{\"final\":\"...\"}\n";
  prompt += "Available tools spec:\n";
  prompt += spec.dump();
  return prompt;
}

static std::string BuildPlannerFinalSystemPrompt() {
  std::string prompt;
  prompt += "You are a tool result summarizer.\n";
  prompt += "You have been given TOOL_RESULT messages.\n";
  prompt += "Return ONLY a single JSON object and no extra text:\n";
  prompt += "{\"final\":\"...\"}\n";
  return prompt;
}

static std::optional<std::string> ExtractFinalFromAssistantJson(const std::string& text) {
  auto j = ParseJsonLoose(text);
  if (!j || !j->is_object()) return std::nullopt;
  if (j->contains("final") && (*j)["final"].is_string()) return (*j)["final"].get<std::string>();
  return std::nullopt;
}

static std::string FakeModelOnce(const std::vector<ChatMessage>& messages) {
  bool has_tool_result = false;
  std::string last_user;
  std::string last_system;
  for (const auto& m : messages) {
    if (m.role == "user") last_user = m.content;
    if (m.role == "system") last_system = m.content;
    if (m.role == "user" && m.content.find("TOOL_RESULT") != std::string::npos) {
      has_tool_result = true;
      break;
    }
  }
  if (last_system.find("You are a planner.") != std::string::npos) {
    if (last_user.find("bad_args") != std::string::npos) {
      return R"({"plan":[{"name":"ide.hover","arguments":{"uri":"file:///Users/acproject/workspace/cpp_projects/local-ai-runtime/src/main.cpp","line":"x","character":2}}]})";
    }
    if (last_user.find("ide.read_file") != std::string::npos) {
      return R"({"plan":[{"name":"ide.read_file","arguments":{"path":"src/main.cpp"}}]})";
    }
    if (last_user.find("ide.search") != std::string::npos) {
      return R"({"plan":[{"name":"ide.search","arguments":{"query":"OpenAiRouter","path":"src"}}]})";
    }
    if (last_user.find("ide.hover") != std::string::npos) {
      return R"({"plan":[{"name":"ide.hover","arguments":{"uri":"file:///Users/acproject/workspace/cpp_projects/local-ai-runtime/src/main.cpp","line":1,"character":2}}]})";
    }
    if (last_user.find("ide.definition") != std::string::npos) {
      return R"({"plan":[{"name":"ide.definition","arguments":{"uri":"file:///Users/acproject/workspace/cpp_projects/local-ai-runtime/src/main.cpp","line":1,"character":2}}]})";
    }
    if (last_user.find("ide.diagnostics") != std::string::npos) {
      return R"({"plan":[{"name":"ide.diagnostics","arguments":{"uri":"file:///Users/acproject/workspace/cpp_projects/local-ai-runtime/src/main.cpp"}}]})";
    }
    if (last_user.find("lsp.hover") != std::string::npos) {
      return R"({"plan":[{"name":"lsp.hover","arguments":{"uri":"file:///Users/acproject/workspace/cpp_projects/local-ai-runtime/src/main.cpp","line":1,"character":2}}]})";
    }
    return R"({"plan":[{"name":"runtime.add","arguments":{"a":2,"b":3}}]})";
  }
  if (last_system.find("tool result summarizer") != std::string::npos) {
    auto pos = last_user.find("TOOL_RESULT");
    if (pos != std::string::npos) return std::string("{\"final\":") + nlohmann::json(last_user).dump() + "}";
    return R"({"final":"done"})";
  }
  if (!has_tool_result) {
    if (last_user.find("mcp2.mcp.echo") != std::string::npos) {
      return R"({"tool_calls":[{"id":"call_1","name":"mcp2.mcp.echo","arguments":{"text":"hello2"}}]})";
    }
    if (last_user.find("mcp.echo") != std::string::npos) {
      return R"({"tool_calls":[{"id":"call_1","name":"mcp.echo","arguments":{"text":"hello"}}]})";
    }
    if (last_user.find("ide.read_file") != std::string::npos) {
      return R"({"tool_calls":[{"id":"call_1","name":"ide.read_file","arguments":{"path":"src/main.cpp"}}]})";
    }
    if (last_user.find("ide.search") != std::string::npos) {
      return R"({"tool_calls":[{"id":"call_1","name":"ide.search","arguments":{"query":"OpenAiRouter","path":"src"}}]})";
    }
    if (last_user.find("ide.hover") != std::string::npos) {
      return R"({"tool_calls":[{"id":"call_1","name":"ide.hover","arguments":{"uri":"file:///Users/acproject/workspace/cpp_projects/local-ai-runtime/src/main.cpp","line":1,"character":2}}]})";
    }
    if (last_user.find("ide.definition") != std::string::npos) {
      return R"({"tool_calls":[{"id":"call_1","name":"ide.definition","arguments":{"uri":"file:///Users/acproject/workspace/cpp_projects/local-ai-runtime/src/main.cpp","line":1,"character":2}}]})";
    }
    if (last_user.find("ide.diagnostics") != std::string::npos) {
      return R"({"tool_calls":[{"id":"call_1","name":"ide.diagnostics","arguments":{"uri":"file:///Users/acproject/workspace/cpp_projects/local-ai-runtime/src/main.cpp"}}]})";
    }
    if (last_user.find("lsp.hover") != std::string::npos) {
      return R"({"tool_calls":[{"id":"call_1","name":"lsp.hover","arguments":{"uri":"file:///Users/acproject/workspace/cpp_projects/local-ai-runtime/src/main.cpp","line":1,"character":2}}]})";
    }
    return R"({"tool_calls":[{"id":"call_1","name":"runtime.add","arguments":{"a":2,"b":3}}]})";
  }
  if (last_user.find("mcp.echo") != std::string::npos || last_user.find("mcp2.mcp.echo") != std::string::npos ||
      last_user.find("lsp.hover") != std::string::npos || last_user.find("ide.hover") != std::string::npos ||
      last_user.find("ide.read_file") != std::string::npos || last_user.find("ide.search") != std::string::npos ||
      last_user.find("ide.definition") != std::string::npos || last_user.find("ide.diagnostics") != std::string::npos) {
    auto pos = last_user.find("TOOL_RESULT");
    if (pos != std::string::npos) {
      return std::string("{\"final\":") + nlohmann::json(last_user).dump() + "}";
    }
    return R"({"final":"done"})";
  }
  return R"({"final":"2 + 3 = 5"})";
}

struct ToolLoopResult {
  std::string final_text;
  std::vector<ToolCall> executed_calls;
  std::vector<ToolResult> results;
  int steps = 0;
  bool hit_step_limit = false;
  bool hit_tool_limit = false;
  bool used_planner = false;
  bool planner_failed = false;
  int plan_steps = 0;
  int plan_rewrites = 0;
  nlohmann::json plan = nlohmann::json::array();
};

static nlohmann::json BuildRuntimeTrace(const ToolLoopResult& loop) {
  nlohmann::json j;
  j["steps"] = loop.steps;
  j["hit_step_limit"] = loop.hit_step_limit;
  j["hit_tool_limit"] = loop.hit_tool_limit;
  j["used_planner"] = loop.used_planner;
  j["planner_failed"] = loop.planner_failed;
  j["plan_steps"] = loop.plan_steps;
  j["plan_rewrites"] = loop.plan_rewrites;
  j["plan"] = loop.plan;
  j["tool_calls"] = nlohmann::json::array();
  for (const auto& c : loop.executed_calls) {
    j["tool_calls"].push_back({{"id", c.id}, {"name", c.name}, {"arguments", c.arguments_json}});
  }
  j["tool_results"] = nlohmann::json::array();
  for (const auto& r : loop.results) {
    j["tool_results"].push_back({{"tool_call_id", r.tool_call_id}, {"name", r.name}, {"ok", r.ok}, {"result", r.result}});
  }
  return j;
}

struct PlannerPlanStep {
  std::string name;
  nlohmann::json arguments;
};

static std::optional<std::vector<PlannerPlanStep>> ParsePlannerPlan(const std::string& assistant_text) {
  auto j = ParseJsonLoose(assistant_text);
  if (!j || !j->is_object()) return std::nullopt;
  if (j->contains("final") && (*j)["final"].is_string()) return std::vector<PlannerPlanStep>{};
  if (!j->contains("plan") || !(*j)["plan"].is_array()) return std::nullopt;
  std::vector<PlannerPlanStep> out;
  for (const auto& s : (*j)["plan"]) {
    if (!s.is_object()) continue;
    if (!s.contains("name") || !s["name"].is_string()) continue;
    PlannerPlanStep step;
    step.name = s["name"].get<std::string>();
    if (s.contains("arguments") && s["arguments"].is_object()) {
      step.arguments = s["arguments"];
    } else {
      step.arguments = nlohmann::json::object();
    }
    if (!step.name.empty()) out.push_back(std::move(step));
  }
  return out;
}

static bool CheckType(const std::string& t, const nlohmann::json& v) {
  if (t == "string") return v.is_string();
  if (t == "integer") return v.is_number_integer();
  if (t == "number") return v.is_number();
  if (t == "boolean") return v.is_boolean();
  if (t == "object") return v.is_object();
  if (t == "array") return v.is_array();
  return true;
}

static bool ValidateSchemaLoose(const nlohmann::json& schema, const nlohmann::json& args, std::string* err) {
  if (!schema.is_object()) return true;
  if (schema.contains("type") && schema["type"].is_string()) {
    auto t = schema["type"].get<std::string>();
    if (!CheckType(t, args)) {
      if (err) *err = "arguments type mismatch";
      return false;
    }
  }
  if (schema.contains("required") && schema["required"].is_array() && args.is_object()) {
    for (const auto& r : schema["required"]) {
      if (!r.is_string()) continue;
      auto k = r.get<std::string>();
      if (!args.contains(k)) {
        if (err) *err = "missing required field: " + k;
        return false;
      }
    }
  }
  if (schema.contains("properties") && schema["properties"].is_object() && args.is_object()) {
    for (const auto& [k, ps] : schema["properties"].items()) {
      if (!args.contains(k)) continue;
      if (!ps.is_object()) continue;
      if (ps.contains("type") && ps["type"].is_string()) {
        if (!CheckType(ps["type"].get<std::string>(), args[k])) {
          if (err) *err = "field type mismatch: " + k;
          return false;
        }
      }
    }
  }
  return true;
}

static std::string ChatOnceText(const std::string& model,
                                const std::vector<ChatMessage>& messages,
                                IProvider* provider,
                                std::string* err) {
  if (model == "fake-tool") return FakeModelOnce(messages);
  ChatRequest req;
  req.model = model;
  req.stream = false;
  req.messages = messages;
  auto resp = provider->ChatOnce(req, err);
  if (!resp) return {};
  return resp->content;
}

static ToolLoopResult RunPlanner(const std::string& model,
                                 const std::vector<ChatMessage>& full_messages,
                                 const std::vector<ToolSchema>& allowed_tools,
                                 const ToolRegistry& registry,
                                 IProvider* provider,
                                 int max_plan_steps,
                                 int max_plan_rewrites,
                                 int max_tool_calls,
                                 std::string* err) {
  ToolLoopResult out;
  out.used_planner = true;

  if (max_plan_steps <= 0) max_plan_steps = 1;
  if (max_plan_rewrites < 0) max_plan_rewrites = 0;
  if (max_tool_calls < 0) max_tool_calls = 0;

  std::unordered_set<std::string> allowed_names;
  for (const auto& t : allowed_tools) allowed_names.insert(t.name);

  std::vector<ChatMessage> plan_msgs;
  plan_msgs.reserve(full_messages.size() + 2);
  plan_msgs.push_back({"system", BuildPlannerSystemPrompt(allowed_tools, max_plan_steps)});
  for (const auto& m : full_messages) plan_msgs.push_back(m);

  std::optional<std::vector<PlannerPlanStep>> plan;
  std::string plan_text;
  int rewrites = 0;
  for (int attempt = 0; attempt <= max_plan_rewrites; attempt++) {
    plan_text = ChatOnceText(model, plan_msgs, provider, err);
    if (plan_text.empty() && err && !err->empty()) {
      out.planner_failed = true;
      return out;
    }
    if (auto final = ExtractFinalFromAssistantJson(plan_text)) {
      out.final_text = *final;
      out.steps = 1;
      out.plan_steps = 0;
      return out;
    }
    plan = ParsePlannerPlan(plan_text);
    if (!plan) {
      if (attempt == max_plan_rewrites) {
        out.planner_failed = true;
        out.final_text = plan_text;
        out.steps = 1;
        return out;
      }
      plan_msgs.push_back({"user", "Plan invalid JSON. Return a corrected plan JSON only."});
      continue;
    }
    bool ok = true;
    std::string why;
    for (const auto& s : *plan) {
      if (!allowed_names.empty() && allowed_names.find(s.name) == allowed_names.end()) {
        ok = false;
        why = "tool not allowed: " + s.name;
        break;
      }
      auto schema = registry.GetSchema(s.name);
      if (!schema) {
        ok = false;
        why = "tool not found: " + s.name;
        break;
      }
      std::string schema_err;
      if (!ValidateSchemaLoose(schema->parameters, s.arguments, &schema_err)) {
        ok = false;
        why = "invalid arguments for " + s.name + ": " + schema_err;
        break;
      }
    }
    if (ok) break;
    if (attempt == max_plan_rewrites) {
      out.planner_failed = true;
      out.final_text = why;
      out.steps = 1;
      return out;
    }
    plan_msgs.push_back({"user", "Plan rejected: " + why + ". Return a corrected plan JSON only."});
    plan.reset();
    rewrites = attempt + 1;
  }

  if (!plan) {
    out.planner_failed = true;
    out.final_text = plan_text;
    out.steps = 1;
    return out;
  }

  if (static_cast<int>(plan->size()) > max_plan_steps) plan->resize(static_cast<size_t>(max_plan_steps));
  out.plan_steps = static_cast<int>(plan->size());
  out.plan_rewrites = rewrites;
  out.plan = nlohmann::json::array();
  for (const auto& s : *plan) out.plan.push_back({{"name", s.name}, {"arguments", s.arguments}});

  std::vector<ChatMessage> exec_msgs;
  exec_msgs.reserve(full_messages.size() + out.plan_steps + 4);
  for (const auto& m : full_messages) exec_msgs.push_back(m);

  int tool_calls_used = 0;
  for (size_t i = 0; i < plan->size(); i++) {
    if (tool_calls_used >= max_tool_calls) {
      out.hit_tool_limit = true;
      out.final_text = "tool call limit exceeded";
      out.steps = static_cast<int>(i + 1);
      return out;
    }

    const auto& s = (*plan)[i];
    ToolCall c;
    c.id = "plan_" + std::to_string(i + 1);
    c.name = s.name;
    c.arguments_json = s.arguments.dump();
    out.executed_calls.push_back(c);

    ToolResult r;
    r.tool_call_id = c.id;
    r.name = c.name;

    if (!allowed_names.empty() && allowed_names.find(c.name) == allowed_names.end()) {
      r.ok = false;
      r.error = "tool not allowed";
      r.result = {{"ok", false}, {"error", r.error}};
      out.results.push_back(r);
      exec_msgs.push_back({"user", "TOOL_RESULT " + c.name + " " + r.result.dump()});
      tool_calls_used++;
      continue;
    }
    auto handler = registry.GetHandler(c.name);
    if (!handler) {
      r.ok = false;
      r.error = "tool not found";
      r.result = {{"ok", false}, {"error", r.error}};
      out.results.push_back(r);
      exec_msgs.push_back({"user", "TOOL_RESULT " + c.name + " " + r.result.dump()});
      tool_calls_used++;
      continue;
    }

    r = (*handler)(c.id, s.arguments);
    out.results.push_back(r);
    exec_msgs.push_back({"user", "TOOL_RESULT " + c.name + " " + r.result.dump()});
    tool_calls_used++;
  }

  std::vector<ChatMessage> final_msgs;
  final_msgs.reserve(exec_msgs.size() + 2);
  final_msgs.push_back({"system", BuildPlannerFinalSystemPrompt()});
  for (const auto& m : exec_msgs) final_msgs.push_back(m);

  auto final_text = ChatOnceText(model, final_msgs, provider, err);
  out.steps = 2;
  if (auto final = ExtractFinalFromAssistantJson(final_text)) {
    out.final_text = *final;
    return out;
  }
  out.final_text = final_text;
  return out;
}

static ToolLoopResult RunToolLoop(const std::string& model,
                                  const std::vector<ChatMessage>& full_messages,
                                  const std::vector<ToolSchema>& allowed_tools,
                                  const ToolRegistry& registry,
                                  IProvider* provider,
                                  int max_steps,
                                  int max_tool_calls,
                                  std::string* err) {
  ToolLoopResult out;
  std::vector<ChatMessage> msgs;
  msgs.reserve(full_messages.size() + 8);

  std::unordered_set<std::string> allowed_names;
  for (const auto& t : allowed_tools) allowed_names.insert(t.name);

  if (!allowed_tools.empty()) {
    msgs.push_back({"system", BuildToolSystemPrompt(allowed_tools)});
  }
  for (const auto& m : full_messages) msgs.push_back(m);

  if (max_steps <= 0) max_steps = 1;
  if (max_tool_calls < 0) max_tool_calls = 0;

  int tool_calls_used = 0;
  for (int step = 0; step < max_steps; step++) {
    out.steps = step + 1;
    std::string assistant_text;
    if (model == "fake-tool") {
      assistant_text = FakeModelOnce(msgs);
    } else {
      ChatRequest req;
      req.model = model;
      req.stream = false;
      req.messages = msgs;
      auto resp = provider->ChatOnce(req, err);
      if (!resp) return out;
      assistant_text = resp->content;
    }

    if (auto calls = ParseToolCallsFromAssistantText(assistant_text)) {
      for (auto& c : *calls) {
        if (!allowed_names.empty() && allowed_names.find(c.name) == allowed_names.end()) {
          ToolResult r;
          r.tool_call_id = c.id;
          r.name = c.name;
          r.ok = false;
          r.error = "tool not allowed";
          r.result = {{"ok", false}, {"error", r.error}};
          out.executed_calls.push_back(c);
          out.results.push_back(r);
          msgs.push_back({"user", "TOOL_RESULT " + c.name + " " + r.result.dump()});
          continue;
        }
        if (!registry.HasTool(c.name)) {
          ToolResult r;
          r.tool_call_id = c.id;
          r.name = c.name;
          r.ok = false;
          r.error = "tool not found";
          r.result = {{"ok", false}, {"error", r.error}};
          out.executed_calls.push_back(c);
          out.results.push_back(r);
          msgs.push_back({"user", "TOOL_RESULT " + c.name + " " + r.result.dump()});
          continue;
        }
        if (tool_calls_used >= max_tool_calls) {
          out.hit_tool_limit = true;
          out.final_text = "tool call limit exceeded";
          return out;
        }
        auto jargs = ParseJsonLoose(c.arguments_json);
        if (!jargs) {
          ToolResult r;
          r.tool_call_id = c.id;
          r.name = c.name;
          r.ok = false;
          r.error = "invalid tool arguments json";
          r.result = {{"ok", false}, {"error", r.error}};
          out.executed_calls.push_back(c);
          out.results.push_back(r);
          msgs.push_back({"user", "TOOL_RESULT " + c.name + " " + r.result.dump()});
          continue;
        }
        auto handler = registry.GetHandler(c.name);
        if (!handler) continue;
        auto r = (*handler)(c.id, *jargs);
        tool_calls_used++;
        out.executed_calls.push_back(c);
        out.results.push_back(r);
        msgs.push_back({"user", "TOOL_RESULT " + c.name + " " + r.result.dump()});
      }
      continue;
    }

    if (auto final = ExtractFinalFromAssistantJson(assistant_text)) {
      out.final_text = *final;
      return out;
    }

    out.final_text = assistant_text;
    return out;
  }

  out.hit_step_limit = true;
  out.final_text = "tool loop exceeded max steps";
  return out;
}

}  // namespace

OpenAiRouter::OpenAiRouter(SessionManager* sessions, ProviderRegistry* providers, ToolRegistry tools)
    : sessions_(sessions), providers_(providers), tools_(std::move(tools)) {}

ToolRegistry* OpenAiRouter::MutableTools() {
  return &tools_;
}

void OpenAiRouter::Register(httplib::Server* server) {
  server->Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
    nlohmann::json out;
    out["object"] = "list";
    out["data"] = nlohmann::json::array();
    const std::string default_provider = providers_ ? providers_->DefaultProviderName() : "";
    if (providers_) {
      for (auto* p : providers_->List()) {
        std::string err;
        auto models = p->ListModels(&err);
        for (const auto& m : models) {
          nlohmann::json item;
          if (p->Name() == default_provider) {
            item["id"] = m.id;
          } else {
            item["id"] = p->Name() + ":" + m.id;
          }
          item["object"] = "model";
          item["created"] = NowSeconds();
          item["owned_by"] = m.owned_by.empty() ? p->Name() : m.owned_by;
          out["data"].push_back(std::move(item));
        }
      }
    }
    SendJson(&res, 200, out);
  });

  server->Post("/v1/embeddings", [&](const httplib::Request& req, httplib::Response& res) {
    auto j = ParseJsonBody(req);
    if (j.is_discarded()) return SendJson(&res, 400, MakeError("invalid json body", "invalid_request_error"));
    if (!j.contains("model") || !j["model"].is_string()) {
      return SendJson(&res, 400, MakeError("missing field: model", "invalid_request_error"));
    }

    std::string model = j["model"].get<std::string>();
    std::string input;
    if (j.contains("input") && j["input"].is_string()) {
      input = j["input"].get<std::string>();
    } else if (j.contains("input") && j["input"].is_array() && !j["input"].empty() && j["input"][0].is_string()) {
      input = j["input"][0].get<std::string>();
    } else {
      return SendJson(&res, 400, MakeError("missing field: input", "invalid_request_error"));
    }

    std::string err;
    auto resolved = providers_ ? providers_->Resolve(model) : std::nullopt;
    if (!resolved) return SendJson(&res, 400, MakeError("unknown provider in model", "invalid_request_error"));
    auto vec = resolved->provider->Embeddings(resolved->model, input, &err);
    if (!vec) return SendJson(&res, 502, MakeError(err.empty() ? "upstream error" : err, "api_error"));

    nlohmann::json out;
    out["object"] = "list";
    out["data"] = nlohmann::json::array();
    nlohmann::json item;
    item["object"] = "embedding";
    item["embedding"] = *vec;
    item["index"] = 0;
    out["data"].push_back(std::move(item));
    out["model"] = model;
    out["usage"] = {{"prompt_tokens", nullptr}, {"total_tokens", nullptr}};
    SendJson(&res, 200, out);
  });

  server->Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
    auto j = ParseJsonBody(req);
    if (j.is_discarded()) return SendJson(&res, 400, MakeError("invalid json body", "invalid_request_error"));
    if (!j.contains("model") || !j["model"].is_string()) {
      return SendJson(&res, 400, MakeError("missing field: model", "invalid_request_error"));
    }

    std::string model = j["model"].get<std::string>();
    bool ok = false;
    auto req_messages = ParseChatMessages(j, &ok);
    if (!ok) return SendJson(&res, 400, MakeError("missing field: messages", "invalid_request_error"));

    std::string session_id;
    if (j.contains("session_id") && j["session_id"].is_string()) session_id = j["session_id"].get<std::string>();
    session_id = sessions_->EnsureSessionId(session_id);
    res.set_header("x-session-id", session_id);

    bool use_server_history = false;
    if (j.contains("use_server_history") && j["use_server_history"].is_boolean()) {
      use_server_history = j["use_server_history"].get<bool>();
    }

    std::vector<ChatMessage> full_messages;
    if (use_server_history) {
      auto sess = sessions_->GetOrCreate(session_id);
      full_messages = sess.history;
      full_messages.insert(full_messages.end(), req_messages.begin(), req_messages.end());
    } else {
      full_messages = req_messages;
    }

    bool stream = false;
    if (j.contains("stream") && j["stream"].is_boolean()) stream = j["stream"].get<bool>();

    int max_steps = 6;
    if (j.contains("max_steps") && j["max_steps"].is_number_integer()) max_steps = j["max_steps"].get<int>();
    int max_tool_calls = 16;
    if (j.contains("max_tool_calls") && j["max_tool_calls"].is_number_integer()) {
      max_tool_calls = j["max_tool_calls"].get<int>();
    }
    bool planner = false;
    int max_plan_steps = 6;
    int max_plan_rewrites = 2;
    if (j.contains("planner")) {
      const auto& p = j["planner"];
      if (p.is_boolean()) {
        planner = p.get<bool>();
      } else if (p.is_object()) {
        if (p.contains("enabled") && p["enabled"].is_boolean()) planner = p["enabled"].get<bool>();
        if (p.contains("max_plan_steps") && p["max_plan_steps"].is_number_integer()) {
          max_plan_steps = p["max_plan_steps"].get<int>();
        }
        if (p.contains("max_rewrites") && p["max_rewrites"].is_number_integer()) {
          max_plan_rewrites = p["max_rewrites"].get<int>();
        }
      }
    }
    bool trace = false;
    if (j.contains("trace") && j["trace"].is_boolean()) trace = j["trace"].get<bool>();

    auto turn_id = NewId("turn");
    TurnRecord turn;
    turn.turn_id = turn_id;
    turn.input_messages = req_messages;

    std::vector<ToolSchema> allowed_tools;
    if (!ToolChoiceIsNone(j)) {
      auto names = ParseRequestedToolNames(j);
      allowed_tools = tools_.FilterSchemas(names);
    }

    IProvider* provider = nullptr;
    std::string provider_model = model;
    if (model != "fake-tool") {
      auto resolved = providers_->Resolve(model);
      if (!resolved) return SendJson(&res, 400, MakeError("unknown provider in model", "invalid_request_error"));
      provider = resolved->provider;
      provider_model = resolved->model;
    }

    if (!stream) {
      std::string err;
      ToolLoopResult loop;
      if (!allowed_tools.empty()) {
        if (planner) {
          loop =
              RunPlanner(provider_model, full_messages, allowed_tools, tools_, provider, max_plan_steps, max_plan_rewrites,
                         max_tool_calls,
                         &err);
          if (loop.planner_failed) {
            loop = RunToolLoop(provider_model, full_messages, allowed_tools, tools_, provider, max_steps, max_tool_calls, &err);
          }
        } else {
          loop = RunToolLoop(provider_model, full_messages, allowed_tools, tools_, provider, max_steps, max_tool_calls, &err);
        }
      } else {
        if (model == "fake-tool") {
          loop.final_text = FakeModelOnce(full_messages);
        } else {
          ChatRequest req;
          req.model = provider_model;
          req.stream = false;
          req.messages = full_messages;
          auto resp = provider->ChatOnce(req, &err);
          if (!resp) return SendJson(&res, 502, MakeError(err.empty() ? "upstream error" : err, "api_error"));
          loop.final_text = resp->content;
        }
      }
      if (loop.final_text.empty() && !err.empty()) return SendJson(&res, 502, MakeError(err, "api_error"));
      if (trace) res.set_header("x-runtime-trace", BuildRuntimeTrace(loop).dump());

      turn.output_text = loop.final_text;
      sessions_->AppendTurn(session_id, turn);
      if (use_server_history) {
        sessions_->AppendToHistory(session_id, req_messages);
        for (const auto& tc : loop.executed_calls) {
          sessions_->AppendToHistory(session_id,
                                     {ChatMessage{"assistant", "TOOL_CALL " + tc.name + " " + tc.arguments_json}});
        }
        for (const auto& tr : loop.results) {
          sessions_->AppendToHistory(session_id, {ChatMessage{"user", "TOOL_RESULT " + tr.name + " " + tr.result.dump()}});
        }
        sessions_->AppendToHistory(session_id, {ChatMessage{"assistant", loop.final_text}});
      }

      nlohmann::json out;
      out["id"] = NewId("chatcmpl");
      out["object"] = "chat.completion";
      out["created"] = NowSeconds();
      out["model"] = model;
      out["choices"] = nlohmann::json::array();
      nlohmann::json choice;
      choice["index"] = 0;
      choice["message"] = {{"role", "assistant"}, {"content", loop.final_text}};
      choice["finish_reason"] = "stop";
      out["choices"].push_back(std::move(choice));
      out["usage"] = {{"prompt_tokens", nullptr}, {"completion_tokens", nullptr}, {"total_tokens", nullptr}};
      return SendJson(&res, 200, out);
    }

    res.status = 200;
    res.set_header("Content-Type", "text/event-stream");
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");
    res.set_header("x-turn-id", turn_id);

    auto id = NewId("chatcmpl");
    auto created = NowSeconds();

    std::string err;
    ToolLoopResult loop;
    if (!allowed_tools.empty()) {
      if (planner) {
        loop =
            RunPlanner(provider_model, full_messages, allowed_tools, tools_, provider, max_plan_steps, max_plan_rewrites,
                       max_tool_calls, &err);
        if (loop.planner_failed) {
          loop = RunToolLoop(provider_model, full_messages, allowed_tools, tools_, provider, max_steps, max_tool_calls, &err);
        }
      } else {
        loop = RunToolLoop(provider_model, full_messages, allowed_tools, tools_, provider, max_steps, max_tool_calls, &err);
      }
    } else {
      if (model == "fake-tool") {
        loop.final_text = FakeModelOnce(full_messages);
      } else {
        ChatRequest req;
        req.model = provider_model;
        req.stream = false;
        req.messages = full_messages;
        auto resp = provider->ChatOnce(req, &err);
        if (resp) loop.final_text = resp->content;
      }
    }

    if (!err.empty() && loop.final_text.empty()) {
      return SendJson(&res, 502, MakeError(err, "api_error"));
    }
    if (trace) res.set_header("x-runtime-trace", BuildRuntimeTrace(loop).dump());

    res.set_chunked_content_provider(
        "text/event-stream",
        [=,
         this,
         turn = std::move(turn),
         loop = std::move(loop)](size_t, httplib::DataSink& sink) mutable {
          std::string acc;
          bool wrote_role = false;

          auto write_chunk = [&](const nlohmann::json& delta, const nlohmann::json& finish_reason) {
            nlohmann::json chunk;
            chunk["id"] = id;
            chunk["object"] = "chat.completion.chunk";
            chunk["created"] = created;
            chunk["model"] = model;
            nlohmann::json choice;
            choice["index"] = 0;
            choice["delta"] = delta;
            choice["finish_reason"] = finish_reason;
            chunk["choices"] = nlohmann::json::array({choice});
            auto s = SseData(chunk);
            sink.write(s.data(), s.size());
          };

          constexpr size_t kArgChunk = 48;
          for (size_t i = 0; i < loop.executed_calls.size(); i++) {
            const auto& c = loop.executed_calls[i];
            std::string args = c.arguments_json.empty() ? "{}" : c.arguments_json;
            for (size_t off = 0; off < args.size(); off += kArgChunk) {
              const bool first_piece = (off == 0);
              nlohmann::json tool_delta;
              if (!wrote_role) {
                tool_delta["role"] = "assistant";
                wrote_role = true;
              }
              nlohmann::json tc;
              tc["index"] = static_cast<int>(i);
              tc["id"] = c.id;
              tc["type"] = "function";
              nlohmann::json func;
              if (first_piece) func["name"] = c.name;
              func["arguments"] = args.substr(off, kArgChunk);
              tc["function"] = std::move(func);
              tool_delta["tool_calls"] = nlohmann::json::array({tc});
              write_chunk(tool_delta, nullptr);
            }
          }

          constexpr size_t kTextChunk = 64;
          for (size_t off = 0; off < loop.final_text.size(); off += kTextChunk) {
            auto piece = loop.final_text.substr(off, kTextChunk);
            nlohmann::json delta;
            if (!wrote_role) {
              delta["role"] = "assistant";
              wrote_role = true;
            }
            delta["content"] = piece;
            write_chunk(delta, nullptr);
            acc += piece;
          }

          write_chunk(nlohmann::json::object(), "stop");

          turn.output_text = acc;
          sessions_->AppendTurn(session_id, turn);
          if (use_server_history) {
            sessions_->AppendToHistory(session_id, turn.input_messages);
            for (const auto& tc : loop.executed_calls) {
              sessions_->AppendToHistory(session_id, {ChatMessage{"assistant", "TOOL_CALL " + tc.name + " " + tc.arguments_json}});
            }
            for (const auto& tr : loop.results) {
              sessions_->AppendToHistory(session_id, {ChatMessage{"user", "TOOL_RESULT " + tr.name + " " + tr.result.dump()}});
            }
            sessions_->AppendToHistory(session_id, {ChatMessage{"assistant", acc}});
          }

          auto done = SseDone();
          sink.write(done.data(), done.size());
          sink.done();
          return true;
        },
        [](bool) {});
  });

  server->Post("/v1/responses", [&](const httplib::Request& req, httplib::Response& res) {
    auto j = ParseJsonBody(req);
    if (j.is_discarded()) return SendJson(&res, 400, MakeError("invalid json body", "invalid_request_error"));
    if (!j.contains("model") || !j["model"].is_string()) {
      return SendJson(&res, 400, MakeError("missing field: model", "invalid_request_error"));
    }

    std::string model = j["model"].get<std::string>();
    std::string input;
    if (j.contains("input") && j["input"].is_string()) {
      input = j["input"].get<std::string>();
    } else if (j.contains("input") && j["input"].is_array() && !j["input"].empty()) {
      const auto& v = j["input"][0];
      if (v.is_string()) input = v.get<std::string>();
      if (v.is_object() && v.contains("content") && v["content"].is_string()) input = v["content"].get<std::string>();
    } else {
      return SendJson(&res, 400, MakeError("missing field: input", "invalid_request_error"));
    }

    std::string err;
    std::string content;
    if (model == "fake-tool") {
      content = FakeModelOnce({ChatMessage{"user", input}});
    } else {
      auto resolved = providers_ ? providers_->Resolve(model) : std::nullopt;
      if (!resolved) return SendJson(&res, 400, MakeError("unknown provider in model", "invalid_request_error"));
      ChatRequest creq;
      creq.model = resolved->model;
      creq.stream = false;
      creq.messages = {ChatMessage{"user", input}};
      auto resp = resolved->provider->ChatOnce(creq, &err);
      if (!resp) return SendJson(&res, 502, MakeError(err.empty() ? "upstream error" : err, "api_error"));
      content = resp->content;
    }

    nlohmann::json out;
    out["id"] = NewId("resp");
    out["object"] = "response";
    out["created"] = NowSeconds();
    out["model"] = model;
    out["output"] = nlohmann::json::array();
    nlohmann::json msg;
    msg["id"] = NewId("msg");
    msg["type"] = "message";
    msg["role"] = "assistant";
    msg["content"] = nlohmann::json::array();
    msg["content"].push_back({{"type", "output_text"}, {"text", content}});
    out["output"].push_back(std::move(msg));
    SendJson(&res, 200, out);
  });
}

}  // namespace runtime
