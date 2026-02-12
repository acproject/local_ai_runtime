#include "tooling.hpp"

#include "config.hpp"
#include "session_manager.hpp"
#include "llama_agent/tool_call_parser.hpp"
#include "llama_agent/tool_manager.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <vector>

namespace runtime {
namespace {

static std::string Trim(const std::string& s) {
  size_t start = 0;
  while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) start++;
  size_t end = s.size();
  while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) end--;
  return s.substr(start, end - start);
}

static std::string ToLower(std::string s) {
  for (auto& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return s;
}

static std::string PercentDecode(const std::string& in) {
  std::string out;
  out.reserve(in.size());
  auto hex = [](char x) -> int {
    if (x >= '0' && x <= '9') return x - '0';
    if (x >= 'a' && x <= 'f') return 10 + (x - 'a');
    if (x >= 'A' && x <= 'F') return 10 + (x - 'A');
    return -1;
  };
  for (size_t i = 0; i < in.size(); i++) {
    const char ch = in[i];
    if (ch == '%' && i + 2 < in.size()) {
      const int ha = hex(in[i + 1]);
      const int hb = hex(in[i + 2]);
      if (ha >= 0 && hb >= 0) {
        out.push_back(static_cast<char>((ha << 4) | hb));
        i += 2;
        continue;
      }
    }
    out.push_back(ch);
  }
  return out;
}

static std::filesystem::path WeakCanonical(const std::filesystem::path& p, std::error_code* ec) {
  auto canon = std::filesystem::weakly_canonical(p, *ec);
  if (*ec) return {};
  return canon;
}

static bool NormalizeUnderRoot(const std::string& workspace_root,
                               const std::string& path_or_uri,
                               std::string* out_path,
                               std::string* err) {
  std::string raw = path_or_uri;
  const std::string lower = ToLower(raw);
  constexpr const char* kFileScheme = "file://";
  if (lower.rfind(kFileScheme, 0) == 0) {
    raw = raw.substr(std::strlen(kFileScheme));
    if (raw.rfind("localhost/", 0) == 0) raw = raw.substr(std::strlen("localhost/"));
    if (!raw.empty() && raw[0] == '/' && raw.size() >= 3 && std::isalpha(static_cast<unsigned char>(raw[1])) && raw[2] == ':') {
      raw = raw.substr(1);
    }
    raw = PercentDecode(raw);
  }

  std::filesystem::path p = raw;
  if (!workspace_root.empty() && p.is_relative()) p = std::filesystem::path(workspace_root) / p;

  std::error_code ec;
  auto canon = WeakCanonical(p, &ec);
  if (ec) {
    if (err) *err = "invalid path";
    return false;
  }

  if (!workspace_root.empty()) {
    auto root = WeakCanonical(std::filesystem::path(workspace_root), &ec);
    if (ec || root.empty()) {
      if (err) *err = "invalid workspace root";
      return false;
    }
    const auto canon_s = canon.generic_string();
    const auto root_s = root.generic_string();
    if (!root_s.empty() && canon_s.rfind(root_s, 0) != 0) {
      if (err) *err = "path is outside workspace root";
      return false;
    }
  }

  if (out_path) *out_path = canon.generic_string();
  return true;
}

static std::string GlobToRegex(const std::string& glob) {
  std::string out;
  out.reserve(glob.size() * 2);
  out += '^';
  for (size_t i = 0; i < glob.size(); i++) {
    const char c = glob[i];
    if (c == '*') {
      const bool is_double = (i + 1 < glob.size() && glob[i + 1] == '*');
      if (is_double) {
        out += ".*";
        i++;
      } else {
        out += "[^/]*";
      }
      continue;
    }
    if (c == '?') {
      out += "[^/]";
      continue;
    }
    if (c == '.') {
      out += "\\.";
      continue;
    }
    if (c == '\\' || c == '/') {
      out += '/';
      continue;
    }
    if (std::string_view("()[]{}+^$|").find(c) != std::string_view::npos) {
      out += '\\';
      out += c;
      continue;
    }
    out += c;
  }
  out += '$';
  return out;
}

static std::vector<std::string> ExpandBraceGlob(const std::string& pattern) {
  auto open = pattern.find('{');
  auto close = pattern.find('}', open == std::string::npos ? 0 : open + 1);
  if (open == std::string::npos || close == std::string::npos || close <= open + 1) return {pattern};
  std::vector<std::string> parts;
  std::string inside = pattern.substr(open + 1, close - open - 1);
  size_t start = 0;
  while (start < inside.size()) {
    size_t comma = inside.find(',', start);
    if (comma == std::string::npos) comma = inside.size();
    parts.push_back(inside.substr(start, comma - start));
    start = comma + 1;
  }
  std::vector<std::string> out;
  out.reserve(parts.size());
  for (const auto& p : parts) {
    out.push_back(pattern.substr(0, open) + p + pattern.substr(close + 1));
  }
  return out;
}

static bool MatchAnyGlob(const std::vector<std::regex>& globs, std::string rel) {
  for (auto& ch : rel) {
    if (ch == '\\') ch = '/';
  }
  if (globs.empty()) return true;
  for (const auto& re : globs) {
    if (std::regex_match(rel, re)) return true;
  }
  return false;
}

static std::optional<std::string> ExtractFirstJsonObject(const std::string& text) {
  auto pos = text.find('{');
  if (pos == std::string::npos) return std::nullopt;
  int depth = 0;
  bool in_string = false;
  bool escape = false;
  for (size_t i = pos; i < text.size(); i++) {
    char c = text[i];
    if (in_string) {
      if (escape) {
        escape = false;
      } else if (c == '\\') {
        escape = true;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }
    if (c == '"') {
      in_string = true;
      continue;
    }
    if (c == '{') depth++;
    if (c == '}') {
      depth--;
      if (depth == 0) return text.substr(pos, i - pos + 1);
    }
  }
  return std::nullopt;
}

static size_t ReplaceAll(std::string* s, const std::string& from, const std::string& to) {
  if (!s) return 0;
  if (from.empty()) return 0;
  size_t count = 0;
  size_t pos = 0;
  while (true) {
    pos = s->find(from, pos);
    if (pos == std::string::npos) break;
    s->replace(pos, from.size(), to);
    pos += to.size();
    count++;
  }
  return count;
}

static std::optional<std::vector<ToolCall>> ExtractToolCallsFromJson(const nlohmann::json& original) {
  if (!original.is_object()) return std::nullopt;

  const nlohmann::json* root = &original;
  if (root->contains("opencode") && (*root)["opencode"].is_object()) root = &(*root)["opencode"];

  auto make_call = [&](const nlohmann::json& item) -> std::optional<ToolCall> {
    if (!item.is_object()) return std::nullopt;
    ToolCall c;
    c.id = NewId("call");
    if (item.contains("id") && item["id"].is_string()) c.id = item["id"].get<std::string>();

    if (item.contains("name") && item["name"].is_string()) c.name = item["name"].get<std::string>();
    if (c.name.empty() && item.contains("tool") && item["tool"].is_string()) c.name = item["tool"].get<std::string>();
    if (c.name.empty() && item.contains("toolName") && item["toolName"].is_string()) c.name = item["toolName"].get<std::string>();
    if (c.name.empty() && item.contains("function") && item["function"].is_object() && item["function"].contains("name") &&
        item["function"]["name"].is_string()) {
      c.name = item["function"]["name"].get<std::string>();
    }

    bool has_args = false;
    auto set_args = [&](const nlohmann::json& a) {
      has_args = true;
      if (a.is_string()) {
        const auto s = a.get<std::string>();
        if (ParseJsonLoose(s)) {
          c.arguments_json = s;
        } else {
          c.arguments_json = nlohmann::json(s).dump();
        }
      } else if (a.is_object() || a.is_array() || a.is_number() || a.is_boolean()) {
        c.arguments_json = a.dump();
      } else if (a.is_null()) {
        c.arguments_json = "{}";
      } else {
        c.arguments_json = a.dump();
      }
    };

    if (item.contains("arguments")) {
      set_args(item["arguments"]);
    } else if (item.contains("args")) {
      set_args(item["args"]);
    } else if (item.contains("input")) {
      set_args(item["input"]);
    } else if (item.contains("function") && item["function"].is_object() && item["function"].contains("arguments")) {
      set_args(item["function"]["arguments"]);
    }

    if (!has_args) return std::nullopt;
    if (c.arguments_json.empty()) c.arguments_json = "{}";
    if (c.name.empty()) return std::nullopt;
    return c;
  };

  for (const auto& key : {"tool_call", "toolCall", "toolcall"}) {
    if (root->contains(key) && (*root)[key].is_object()) {
      if (auto c = make_call((*root)[key])) return std::vector<ToolCall>{*c};
    }
  }

  if (auto c = make_call(*root)) return std::vector<ToolCall>{*c};

  const nlohmann::json* tool_calls = nullptr;
  for (const auto& key : {"tool_calls", "toolCalls", "toolcalls"}) {
    if (root->contains(key) && (*root)[key].is_array()) {
      tool_calls = &(*root)[key];
      break;
    }
  }
  if (!tool_calls) return std::nullopt;

  std::vector<ToolCall> calls;
  for (size_t i = 0; i < tool_calls->size(); i++) {
    if (auto c = make_call((*tool_calls)[i])) calls.push_back(std::move(*c));
  }
  if (calls.empty()) return std::nullopt;
  return calls;
}

static bool IsToolNameChar(char ch) {
  const unsigned char c = static_cast<unsigned char>(ch);
  return std::isalnum(c) || ch == '_' || ch == '-' || ch == '.' || ch == ':' || ch == '/';
}

static std::optional<std::vector<ToolCall>> ExtractToolCallsFromTaggedText(const std::string& assistant_text) {
  std::string lower;
  lower.reserve(assistant_text.size());
  for (char ch : assistant_text) lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));

  const std::string tool_tag = "<tool_call";
  const std::string tool_tag2 = "<toolcall";
  const std::string arg_tag = "<arg_value>";
  const std::string arg_end = "</arg_value>";
  const std::string arg_key_end = "</arg_key>";

  std::vector<ToolCall> calls;
  size_t pos = 0;
  while (pos < lower.size()) {
    size_t start = lower.find(tool_tag, pos);
    if (start == std::string::npos) start = lower.find(tool_tag2, pos);
    if (start == std::string::npos) break;

    size_t tag_close = lower.find('>', start);
    if (tag_close == std::string::npos) break;

    std::string name;
    std::string tag_text = assistant_text.substr(start, tag_close - start + 1);
    std::string tag_lower = lower.substr(start, tag_close - start + 1);
    auto find_attr = [&](const std::string& attr) -> std::optional<std::string> {
      auto p = tag_lower.find(attr);
      if (p == std::string::npos) return std::nullopt;
      p += attr.size();
      while (p < tag_text.size() && std::isspace(static_cast<unsigned char>(tag_text[p]))) p++;
      if (p >= tag_text.size() || tag_text[p] != '=') return std::nullopt;
      p++;
      while (p < tag_text.size() && std::isspace(static_cast<unsigned char>(tag_text[p]))) p++;
      if (p >= tag_text.size()) return std::nullopt;
      if (tag_text[p] == '"' || tag_text[p] == '\'') {
        const char q = tag_text[p++];
        size_t qend = tag_text.find(q, p);
        if (qend == std::string::npos) return std::nullopt;
        return tag_text.substr(p, qend - p);
      }
      size_t e = p;
      while (e < tag_text.size() && !std::isspace(static_cast<unsigned char>(tag_text[e])) && tag_text[e] != '>') e++;
      if (e <= p) return std::nullopt;
      return tag_text.substr(p, e - p);
    };

    if (auto n = find_attr("name")) name = Trim(*n);
    size_t after_name = tag_close + 1;
    if (name.empty()) {
      size_t name_start = tag_close + 1;
      while (name_start < assistant_text.size() && std::isspace(static_cast<unsigned char>(assistant_text[name_start]))) name_start++;
      size_t name_end = name_start;
      while (name_end < assistant_text.size() && IsToolNameChar(assistant_text[name_end])) name_end++;
      name = Trim(assistant_text.substr(name_start, name_end - name_start));
      after_name = name_end;
    }

    if (name.empty()) {
      pos = tag_close + 1;
      continue;
    }

    size_t block_start = tag_close + 1;
    size_t next_tool = lower.find(tool_tag, block_start);
    size_t next_tool2 = lower.find(tool_tag2, block_start);
    if (next_tool == std::string::npos || (next_tool2 != std::string::npos && next_tool2 < next_tool)) next_tool = next_tool2;
    size_t block_end = (next_tool == std::string::npos) ? assistant_text.size() : next_tool;

    std::string args_text;
    size_t astart = lower.find(arg_tag, after_name);
    if (astart != std::string::npos && astart < block_end) {
      astart += arg_tag.size();
      size_t aend = lower.find(arg_end, astart);
      if (aend == std::string::npos || aend > block_end) aend = block_end;
      args_text = Trim(assistant_text.substr(astart, aend - astart));
    } else {
      size_t maybe_close = lower.find(arg_end, after_name);
      if (maybe_close != std::string::npos && maybe_close < block_end) {
        size_t raw_start = after_name;
        size_t key_close = lower.rfind(arg_key_end, maybe_close);
        if (key_close != std::string::npos && key_close >= after_name) {
          raw_start = key_close + arg_key_end.size();
        }
        if (raw_start <= maybe_close) args_text = Trim(assistant_text.substr(raw_start, maybe_close - raw_start));
        if (args_text.empty()) {
          size_t raw2 = maybe_close + arg_end.size();
          if (raw2 < block_end) args_text = Trim(assistant_text.substr(raw2, block_end - raw2));
        }
      } else {
        args_text = Trim(assistant_text.substr(after_name, block_end - after_name));
      }
    }

    if (!args_text.empty()) {
      if (auto first = ExtractFirstJsonObject(args_text)) args_text = Trim(*first);
    }

    ToolCall c;
    c.id = NewId("call");
    c.name = name;
    if (!args_text.empty()) {
      auto j = ParseJsonLoose(args_text);
      if (j) {
        c.arguments_json = j->dump();
      } else {
        auto raw = Trim(args_text);
        if (auto lt = raw.find('<'); lt != std::string::npos) raw = Trim(raw.substr(0, lt));
        if (!raw.empty() && c.name == "cat") {
          auto raw_lower = raw;
          for (auto& ch : raw_lower) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
          if (raw_lower.rfind("cat", 0) == 0) {
            size_t p = 3;
            while (p < raw.size() && std::isspace(static_cast<unsigned char>(raw[p]))) p++;
            if (p < raw.size()) raw = Trim(raw.substr(p));
          }
          if (!raw.empty() && raw.front() == '`') raw = Trim(raw.substr(1));
          while (!raw.empty() && (raw.back() == '`' || raw.back() == ';')) raw.pop_back();
          raw = Trim(raw);
        }
        c.arguments_json = nlohmann::json(raw).dump();
      }
    } else {
      c.arguments_json = "{}";
    }

    if (c.name == "cat") c.name = "read";
    calls.push_back(std::move(c));

    pos = block_end;
  }

  if (calls.empty()) return std::nullopt;
  return calls;
}

static std::optional<std::string> ExtractBalanced(const std::string& text, size_t start) {
  if (start >= text.size()) return std::nullopt;
  const char open = text[start];
  const char close = (open == '{') ? '}' : ((open == '[') ? ']' : 0);
  if (close == 0) return std::nullopt;

  int depth = 0;
  bool in_string = false;
  bool escape = false;
  for (size_t i = start; i < text.size(); i++) {
    const char c = text[i];
    if (in_string) {
      if (escape) {
        escape = false;
      } else if (c == '\\') {
        escape = true;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }
    if (c == '"') {
      in_string = true;
      continue;
    }
    if (c == open) depth++;
    if (c == close) {
      depth--;
      if (depth == 0) return text.substr(start, i - start + 1);
    }
  }
  return std::nullopt;
}

static std::optional<std::vector<ToolCall>> ExtractToolCallsFromCommandText(const std::string& assistant_text) {
  std::string lower;
  lower.reserve(assistant_text.size());
  for (char ch : assistant_text) lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));

  const std::string tool = "todowrite";
  std::vector<ToolCall> calls;

  size_t pos = 0;
  while (pos < lower.size()) {
    size_t start = lower.find(tool, pos);
    if (start == std::string::npos) break;

    const bool left_ok = (start == 0) || std::isspace(static_cast<unsigned char>(lower[start - 1])) || lower[start - 1] == '`';
    const size_t after = start + tool.size();
    const bool right_ok = (after >= lower.size()) || std::isspace(static_cast<unsigned char>(lower[after])) || lower[after] == ':' ||
                          lower[after] == '(';
    if (!left_ok || !right_ok) {
      pos = after;
      continue;
    }

    size_t args_start = after;
    if (args_start < assistant_text.size() && assistant_text[args_start] == ':') args_start++;

    nlohmann::json args = nlohmann::json::object();
    bool added_call = false;
    size_t p = args_start;
    while (p < assistant_text.size()) {
      while (p < assistant_text.size() &&
             (std::isspace(static_cast<unsigned char>(assistant_text[p])) || assistant_text[p] == ',' || assistant_text[p] == ';')) {
        p++;
      }
      if (p >= assistant_text.size()) break;
      if (assistant_text[p] == '{') {
        if (auto obj = ExtractBalanced(assistant_text, p)) {
          auto j = nlohmann::json::parse(*obj, nullptr, false);
          if (!j.is_discarded() && j.is_object()) {
            ToolCall c;
            c.id = NewId("call");
            c.name = tool;
            c.arguments_json = j.dump();
            calls.push_back(std::move(c));
            added_call = true;
            break;
          }
        }
        break;
      }

      size_t key_start = p;
      while (p < assistant_text.size()) {
        const unsigned char ch = static_cast<unsigned char>(assistant_text[p]);
        if (!(std::isalnum(ch) || assistant_text[p] == '_')) break;
        p++;
      }
      if (p <= key_start) break;
      std::string key = assistant_text.substr(key_start, p - key_start);

      while (p < assistant_text.size() && std::isspace(static_cast<unsigned char>(assistant_text[p]))) p++;
      if (p >= assistant_text.size() || assistant_text[p] != '=') break;
      p++;
      while (p < assistant_text.size() && std::isspace(static_cast<unsigned char>(assistant_text[p]))) p++;
      if (p >= assistant_text.size()) break;

      std::string raw_value;
      if (assistant_text[p] == '"' || assistant_text[p] == '\'') {
        const char q = assistant_text[p++];
        size_t vstart = p;
        bool esc = false;
        for (; p < assistant_text.size(); p++) {
          const char c = assistant_text[p];
          if (esc) {
            esc = false;
            continue;
          }
          if (c == '\\') {
            esc = true;
            continue;
          }
          if (c == q) break;
        }
        raw_value = assistant_text.substr(vstart, (p > vstart ? p - vstart : 0));
        if (p < assistant_text.size() && assistant_text[p] == q) p++;
      } else if (assistant_text[p] == '{' || assistant_text[p] == '[') {
        if (auto b = ExtractBalanced(assistant_text, p)) {
          raw_value = *b;
          p += b->size();
        } else {
          break;
        }
      } else {
        size_t vstart = p;
        while (p < assistant_text.size() && !std::isspace(static_cast<unsigned char>(assistant_text[p])) && assistant_text[p] != ',' &&
               assistant_text[p] != ';') {
          p++;
        }
        raw_value = assistant_text.substr(vstart, p - vstart);
      }

      auto trimmed = Trim(raw_value);
      if (!trimmed.empty() && (trimmed[0] == '{' || trimmed[0] == '[')) {
        auto j = nlohmann::json::parse(trimmed, nullptr, false);
        if (!j.is_discarded()) {
          args[key] = std::move(j);
        } else {
          args[key] = trimmed;
        }
      } else {
        args[key] = trimmed;
      }
    }

    if (!calls.empty() && calls.back().name == tool) {
      pos = after;
      continue;
    }

    if (!args.empty()) {
      ToolCall c;
      c.id = NewId("call");
      c.name = tool;
      c.arguments_json = args.dump();
      calls.push_back(std::move(c));
    }

    pos = after;
  }

  if (calls.empty()) return std::nullopt;
  return calls;
}

static std::optional<std::vector<ToolCall>> ExtractToolCallsFromCatCommandText(const std::string& assistant_text) {
  std::string lower;
  lower.reserve(assistant_text.size());
  for (char ch : assistant_text) lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));

  const std::string cmd = "cat";
  std::vector<ToolCall> calls;

  size_t pos = 0;
  while (pos < lower.size()) {
    size_t start = lower.find(cmd, pos);
    if (start == std::string::npos) break;

    const bool left_ok = (start == 0) || std::isspace(static_cast<unsigned char>(lower[start - 1])) || lower[start - 1] == '`' ||
                         lower[start - 1] == ':';
    const size_t after = start + cmd.size();
    const bool right_ok = (after >= lower.size()) || std::isspace(static_cast<unsigned char>(lower[after])) || lower[after] == '`';
    if (!left_ok || !right_ok) {
      pos = after;
      continue;
    }

    size_t p = after;
    while (p < assistant_text.size() && std::isspace(static_cast<unsigned char>(assistant_text[p]))) p++;
    if (p >= assistant_text.size()) {
      pos = after;
      continue;
    }

    std::string raw_path;
    if (assistant_text[p] == '"' || assistant_text[p] == '\'') {
      const char q = assistant_text[p++];
      size_t vstart = p;
      bool esc = false;
      for (; p < assistant_text.size(); p++) {
        const char c = assistant_text[p];
        if (esc) {
          esc = false;
          continue;
        }
        if (c == '\\') {
          esc = true;
          continue;
        }
        if (c == q) break;
      }
      raw_path = assistant_text.substr(vstart, (p > vstart ? p - vstart : 0));
      if (p < assistant_text.size() && assistant_text[p] == q) p++;
    } else {
      size_t vstart = p;
      while (p < assistant_text.size() && !std::isspace(static_cast<unsigned char>(assistant_text[p])) && assistant_text[p] != ';' &&
             assistant_text[p] != ',' && assistant_text[p] != '<' && assistant_text[p] != '`') {
        p++;
      }
      raw_path = assistant_text.substr(vstart, p - vstart);
    }

    auto path = Trim(raw_path);
    if (auto lt = path.find('<'); lt != std::string::npos) path = Trim(path.substr(0, lt));
    while (!path.empty() && (path.back() == '`' || path.back() == ';' || path.back() == ',')) path.pop_back();
    path = Trim(path);

    if (!path.empty()) {
      ToolCall c;
      c.id = NewId("call");
      c.name = "read";
      nlohmann::json args = nlohmann::json::object();
      args["filePath"] = path;
      c.arguments_json = args.dump();
      calls.push_back(std::move(c));
    }

    pos = after;
  }

  if (calls.empty()) return std::nullopt;
  return calls;
}

static nlohmann::json ErrorResult(const std::string& message) {
  nlohmann::json j;
  j["ok"] = false;
  j["error"] = message;
  return j;
}

static llama_agent::ToolDefinition ToAgentToolDefinition(const ToolSchema& schema) {
  llama_agent::ToolDefinition def;
  def.name = schema.name;
  def.description = schema.description;
  def.jsonSchema = {{"name", schema.name}, {"description", schema.description}, {"parameters", schema.parameters}};

  if (schema.parameters.is_object()) {
    std::unordered_set<std::string> required;
    if (schema.parameters.contains("required") && schema.parameters["required"].is_array()) {
      for (const auto& it : schema.parameters["required"]) {
        if (it.is_string()) required.insert(it.get<std::string>());
      }
    }

    if (schema.parameters.contains("properties") && schema.parameters["properties"].is_object()) {
      for (const auto& [name, prop] : schema.parameters["properties"].items()) {
        llama_agent::ToolParameter p;
        p.name = name;
        p.required = required.find(name) != required.end();
        p.schema = prop;
        if (prop.is_object()) {
          if (prop.contains("type") && prop["type"].is_string()) p.type = prop["type"].get<std::string>();
          if (prop.contains("description") && prop["description"].is_string()) p.description = prop["description"].get<std::string>();
        }
        def.parameters.push_back(std::move(p));
      }
    }
  }

  return def;
}

}  // namespace

ToolRegistry::~ToolRegistry() = default;

ToolRegistry::ToolRegistry(ToolRegistry&& other) noexcept {
  std::unique_lock<std::shared_mutex> lock(other.mu_);
  schemas_ = std::move(other.schemas_);
  handlers_ = std::move(other.handlers_);
  tool_manager_ = std::move(other.tool_manager_);
}

ToolRegistry& ToolRegistry::operator=(ToolRegistry&& other) noexcept {
  if (this == &other) return *this;
  std::unique_lock<std::shared_mutex> lock_other(other.mu_);
  std::unique_lock<std::shared_mutex> lock_this(mu_);
  schemas_ = std::move(other.schemas_);
  handlers_ = std::move(other.handlers_);
  tool_manager_ = std::move(other.tool_manager_);
  return *this;
}

void ToolRegistry::RegisterTool(ToolSchema schema, ToolHandler handler) {
  std::unique_lock<std::shared_mutex> lock(mu_);
  const auto name = schema.name;
  schemas_[name] = std::move(schema);
  handlers_[name] = handler;

  if (!tool_manager_) tool_manager_ = std::make_unique<llama_agent::ToolManager>();
  auto def = ToAgentToolDefinition(schemas_[name]);
  tool_manager_->registerTool(def, [handler = std::move(handler), name](const nlohmann::json& arguments) mutable {
    auto r = handler("call_0", arguments);
    nlohmann::json out = r.result;
    if (!out.is_object()) {
      nlohmann::json wrap;
      wrap["ok"] = r.ok;
      wrap["result"] = out;
      if (!r.ok && !r.error.empty()) wrap["error"] = r.error;
      out = std::move(wrap);
    } else {
      if (!out.contains("ok")) out["ok"] = r.ok;
      if (!r.ok && !out.contains("error") && !r.error.empty()) out["error"] = r.error;
    }
    return out;
  });
}

bool ToolRegistry::HasTool(const std::string& name) const {
  std::shared_lock<std::shared_mutex> lock(mu_);
  if (tool_manager_) return tool_manager_->hasTool(name);
  return schemas_.find(name) != schemas_.end() && handlers_.find(name) != handlers_.end();
}

std::optional<ToolSchema> ToolRegistry::GetSchema(const std::string& name) const {
  std::shared_lock<std::shared_mutex> lock(mu_);
  auto it = schemas_.find(name);
  if (it == schemas_.end()) return std::nullopt;
  return it->second;
}

std::optional<ToolHandler> ToolRegistry::GetHandler(const std::string& name) const {
  std::shared_lock<std::shared_mutex> lock(mu_);
  auto it = handlers_.find(name);
  if (it == handlers_.end()) return std::nullopt;
  return it->second;
}

std::vector<ToolSchema> ToolRegistry::ListSchemas() const {
  std::shared_lock<std::shared_mutex> lock(mu_);
  std::vector<ToolSchema> out;
  out.reserve(schemas_.size());
  for (const auto& [_, schema] : schemas_) out.push_back(schema);
  return out;
}

std::vector<ToolSchema> ToolRegistry::FilterSchemas(const std::vector<std::string>& allow_names) const {
  std::shared_lock<std::shared_mutex> lock(mu_);
  std::vector<ToolSchema> out;
  out.reserve(allow_names.size());
  for (const auto& name : allow_names) {
    auto it = schemas_.find(name);
    if (it != schemas_.end()) out.push_back(it->second);
  }
  return out;
}

ToolRegistry BuildDefaultToolRegistry(const RuntimeConfig& cfg) {
  ToolRegistry reg;

  std::string workspace_root = cfg.workspace_root;
  if (workspace_root.empty()) {
    std::error_code ec;
    workspace_root = std::filesystem::current_path(ec).generic_string();
  }
  {
    std::error_code ec;
    auto canon = WeakCanonical(std::filesystem::path(workspace_root), &ec);
    if (!ec && !canon.empty()) workspace_root = canon.generic_string();
  }

  {
    ToolSchema schema;
    schema.name = "runtime.echo";
    schema.description = "Echo back the provided text.";
    schema.parameters = {{"type", "object"},
                         {"properties", {{"text", {{"type", "string"}}}}},
                         {"required", {"text"}}};
    reg.RegisterTool(schema, [](const std::string& tool_call_id, const nlohmann::json& arguments) {
      ToolResult r;
      r.tool_call_id = tool_call_id;
      r.name = "runtime.echo";
      if (!arguments.is_object() || !arguments.contains("text") || !arguments["text"].is_string()) {
        r.ok = false;
        r.error = "missing required field: text";
        r.result = ErrorResult(r.error);
        return r;
      }
      r.result = {{"ok", true}, {"text", arguments["text"].get<std::string>()}};
      return r;
    });
  }

  {
    ToolSchema schema;
    schema.name = "read";
    schema.description = "Read a text file.";
    schema.parameters = {{"type", "object"},
                         {"properties",
                          {{"filePath", {{"type", "string"}}}, {"offset", {{"type", "integer"}}}, {"limit", {{"type", "integer"}}}}},
                         {"required", {"filePath"}}};
    auto handler = [workspace_root, schema](const std::string& tool_call_id, const nlohmann::json& arguments) {
      constexpr int kDefaultLimit = 2000;
      constexpr size_t kMaxLineLength = 2000;
      constexpr size_t kMaxBytes = 50 * 1024;

      ToolResult r;
      r.tool_call_id = tool_call_id;
      r.name = schema.name;

      if (!arguments.is_object() || !arguments.contains("filePath") || !arguments["filePath"].is_string()) {
        r.ok = false;
        r.error = "missing required field: filePath";
        r.result = ErrorResult(r.error);
        return r;
      }

      int offset = 0;
      int limit = kDefaultLimit;
      if (arguments.contains("offset") && arguments["offset"].is_number_integer()) offset = arguments["offset"].get<int>();
      if (arguments.contains("limit") && arguments["limit"].is_number_integer()) limit = arguments["limit"].get<int>();
      if (offset < 0) offset = 0;
      if (limit <= 0) limit = kDefaultLimit;

      std::string norm, err;
      if (!NormalizeUnderRoot(workspace_root, arguments["filePath"].get<std::string>(), &norm, &err)) {
        r.ok = false;
        r.error = err.empty() ? "invalid path" : err;
        r.result = ErrorResult(r.error);
        return r;
      }

      std::ifstream in(norm, std::ios::binary);
      if (!in) {
        r.ok = false;
        r.error = "file not found";
        r.result = ErrorResult(r.error);
        return r;
      }

      std::vector<std::string> out_lines;
      out_lines.reserve(static_cast<size_t>(std::min(limit, 2000)));
      std::string line;
      int idx = 0;
      int total_lines = 0;
      size_t bytes = 0;
      bool truncated_by_bytes = false;

      while (std::getline(in, line)) {
        total_lines++;
        if (idx < offset) {
          idx++;
          continue;
        }
        if (static_cast<int>(out_lines.size()) >= limit) {
          idx++;
          continue;
        }
        std::string shown = line;
        if (shown.size() > kMaxLineLength) {
          shown.resize(kMaxLineLength);
          shown += "...";
        }
        size_t add = shown.size() + (out_lines.empty() ? 0 : 1);
        if (bytes + add > kMaxBytes) {
          truncated_by_bytes = true;
          break;
        }
        bytes += add;
        out_lines.push_back(std::move(shown));
        idx++;
      }

      const int last_read_line = offset + static_cast<int>(out_lines.size());
      const bool has_more_lines = total_lines > last_read_line || (!in.eof() && !truncated_by_bytes);
      const bool truncated = has_more_lines || truncated_by_bytes;

      std::ostringstream oss;
      oss << "<file>\n";
      for (size_t i = 0; i < out_lines.size(); i++) {
        const int n = offset + static_cast<int>(i) + 1;
        oss << std::setw(5) << std::setfill('0') << n << "| " << out_lines[i];
        if (i + 1 < out_lines.size()) oss << "\n";
      }
      if (truncated_by_bytes) {
        oss << "\n\n(Output truncated at " << kMaxBytes << " bytes. Use 'offset' parameter to read beyond line " << last_read_line << ")";
      } else if (has_more_lines) {
        oss << "\n\n(File has more lines. Use 'offset' parameter to read beyond line " << last_read_line << ")";
      } else {
        oss << "\n\n(End of file - total " << total_lines << " lines)";
      }
      oss << "\n</file>";

      r.result = {{"ok", true},
                  {"title", norm},
                  {"output", oss.str()},
                  {"metadata", {{"truncated", truncated}, {"lastReadLine", last_read_line}, {"totalLines", total_lines}}}};
      return r;
    };
    reg.RegisterTool(schema, handler);

    ToolSchema s2 = schema;
    s2.name = "readFile";
    reg.RegisterTool(s2, [handler, s2](const std::string& tool_call_id, const nlohmann::json& arguments) mutable {
      auto r = handler(tool_call_id, arguments);
      r.name = s2.name;
      return r;
    });

    ToolSchema s3 = schema;
    s3.name = "read_file";
    reg.RegisterTool(s3, [handler, s3](const std::string& tool_call_id, const nlohmann::json& arguments) mutable {
      auto r = handler(tool_call_id, arguments);
      r.name = s3.name;
      return r;
    });
  }

  {
    ToolSchema schema;
    schema.name = "write";
    schema.description = "Write text content to a file.";
    schema.parameters = {{"type", "object"},
                         {"properties", {{"content", {{"type", "string"}}}, {"filePath", {{"type", "string"}}}}},
                         {"required", {"content", "filePath"}}};
    auto handler = [workspace_root, schema](const std::string& tool_call_id, const nlohmann::json& arguments) {
      ToolResult r;
      r.tool_call_id = tool_call_id;
      r.name = schema.name;

      if (!arguments.is_object() || !arguments.contains("filePath") || !arguments["filePath"].is_string() ||
          !arguments.contains("content") || !arguments["content"].is_string()) {
        r.ok = false;
        r.error = "missing required fields: filePath, content";
        r.result = ErrorResult(r.error);
        return r;
      }

      std::string norm, err;
      if (!NormalizeUnderRoot(workspace_root, arguments["filePath"].get<std::string>(), &norm, &err)) {
        r.ok = false;
        r.error = err.empty() ? "invalid path" : err;
        r.result = ErrorResult(r.error);
        return r;
      }

      std::error_code ec;
      const bool existed = std::filesystem::exists(std::filesystem::path(norm), ec) && !ec;
      auto parent = std::filesystem::path(norm).parent_path();
      if (!parent.empty()) std::filesystem::create_directories(parent, ec);

      std::ofstream out(norm, std::ios::binary | std::ios::trunc);
      if (!out) {
        r.ok = false;
        r.error = "failed to open file for writing";
        r.result = ErrorResult(r.error);
        return r;
      }
      const auto& content = arguments["content"].get<std::string>();
      out.write(content.data(), static_cast<std::streamsize>(content.size()));
      out.close();

      r.result = {{"ok", true}, {"title", norm}, {"output", ""}, {"metadata", {{"filepath", norm}, {"exists", existed}}}};
      return r;
    };
    reg.RegisterTool(schema, handler);

    ToolSchema s2 = schema;
    s2.name = "writeFile";
    reg.RegisterTool(s2, [handler, s2](const std::string& tool_call_id, const nlohmann::json& arguments) mutable {
      auto r = handler(tool_call_id, arguments);
      r.name = s2.name;
      return r;
    });
  }

  {
    ToolSchema schema;
    schema.name = "edit";
    schema.description = "Edit a file by replacing a string.";
    schema.parameters = {{"type", "object"},
                         {"properties",
                          {{"filePath", {{"type", "string"}}},
                           {"oldString", {{"type", "string"}}},
                           {"newString", {{"type", "string"}}},
                           {"replaceAll", {{"type", "boolean"}}}}},
                         {"required", {"filePath", "oldString", "newString"}}};
    auto handler = [workspace_root, schema](const std::string& tool_call_id, const nlohmann::json& arguments) {
      ToolResult r;
      r.tool_call_id = tool_call_id;
      r.name = schema.name;

      if (!arguments.is_object() || !arguments.contains("filePath") || !arguments["filePath"].is_string() ||
          !arguments.contains("oldString") || !arguments["oldString"].is_string() || !arguments.contains("newString") ||
          !arguments["newString"].is_string()) {
        r.ok = false;
        r.error = "missing required fields: filePath, oldString, newString";
        r.result = ErrorResult(r.error);
        return r;
      }

      const std::string old_string = arguments["oldString"].get<std::string>();
      const std::string new_string = arguments["newString"].get<std::string>();
      if (old_string == new_string) {
        r.ok = false;
        r.error = "oldString and newString must be different";
        r.result = ErrorResult(r.error);
        return r;
      }

      bool replace_all = false;
      if (arguments.contains("replaceAll") && arguments["replaceAll"].is_boolean()) replace_all = arguments["replaceAll"].get<bool>();

      std::string norm, err;
      if (!NormalizeUnderRoot(workspace_root, arguments["filePath"].get<std::string>(), &norm, &err)) {
        r.ok = false;
        r.error = err.empty() ? "invalid path" : err;
        r.result = ErrorResult(r.error);
        return r;
      }

      std::error_code ec;
      auto parent = std::filesystem::path(norm).parent_path();
      if (!parent.empty()) std::filesystem::create_directories(parent, ec);

      if (old_string.empty()) {
        std::ofstream out(norm, std::ios::binary | std::ios::trunc);
        if (!out) {
          r.ok = false;
          r.error = "failed to open file for writing";
          r.result = ErrorResult(r.error);
          return r;
        }
        out.write(new_string.data(), static_cast<std::streamsize>(new_string.size()));
        out.close();
        r.result = {{"ok", true}, {"title", norm}, {"output", ""}, {"metadata", {{"filepath", norm}}}};
        return r;
      }

      std::ifstream in(norm, std::ios::binary);
      if (!in) {
        r.ok = false;
        r.error = "file not found";
        r.result = ErrorResult(r.error);
        return r;
      }
      std::ostringstream oss;
      oss << in.rdbuf();
      std::string content = oss.str();
      in.close();

      auto first = content.find(old_string);
      if (first == std::string::npos) {
        r.ok = false;
        r.error = "oldString not found in content";
        r.result = ErrorResult(r.error);
        return r;
      }

      size_t replacements = 0;
      if (replace_all) {
        replacements = ReplaceAll(&content, old_string, new_string);
      } else {
        auto last = content.rfind(old_string);
        if (last != first) {
          r.ok = false;
          r.error = "found multiple matches for oldString; set replaceAll=true or provide a more specific oldString";
          r.result = ErrorResult(r.error);
          return r;
        }
        content.replace(first, old_string.size(), new_string);
        replacements = 1;
      }

      std::ofstream out(norm, std::ios::binary | std::ios::trunc);
      if (!out) {
        r.ok = false;
        r.error = "failed to open file for writing";
        r.result = ErrorResult(r.error);
        return r;
      }
      out.write(content.data(), static_cast<std::streamsize>(content.size()));
      out.close();

      r.result = {{"ok", true}, {"title", norm}, {"output", ""}, {"metadata", {{"filepath", norm}, {"replacements", replacements}}}};
      return r;
    };
    reg.RegisterTool(schema, handler);

    ToolSchema s2 = schema;
    s2.name = "editFile";
    reg.RegisterTool(s2, [handler, s2](const std::string& tool_call_id, const nlohmann::json& arguments) mutable {
      auto r = handler(tool_call_id, arguments);
      r.name = s2.name;
      return r;
    });
  }

  {
    ToolSchema schema;
    schema.name = "glob";
    schema.description = "Match files using a glob pattern.";
    schema.parameters = {{"type", "object"},
                         {"properties", {{"pattern", {{"type", "string"}}}, {"path", {{"type", "string"}}}}},
                         {"required", {"pattern"}}};
    reg.RegisterTool(schema, [workspace_root, schema](const std::string& tool_call_id, const nlohmann::json& arguments) {
      ToolResult r;
      r.tool_call_id = tool_call_id;
      r.name = schema.name;
      if (!arguments.is_object() || !arguments.contains("pattern") || !arguments["pattern"].is_string()) {
        r.ok = false;
        r.error = "missing required field: pattern";
        r.result = ErrorResult(r.error);
        return r;
      }

      std::string base = ".";
      if (arguments.contains("path") && arguments["path"].is_string()) base = arguments["path"].get<std::string>();

      std::string norm_base, err;
      if (!NormalizeUnderRoot(workspace_root, base, &norm_base, &err)) {
        r.ok = false;
        r.error = err.empty() ? "invalid path" : err;
        r.result = ErrorResult(r.error);
        return r;
      }

      const std::string pattern = arguments["pattern"].get<std::string>();
      std::vector<std::regex> globs;
      try {
        for (const auto& p : ExpandBraceGlob(pattern)) globs.emplace_back(GlobToRegex(p), std::regex::ECMAScript);
      } catch (const std::exception& e) {
        r.ok = false;
        r.error = std::string("invalid glob pattern: ") + e.what();
        r.result = ErrorResult(r.error);
        return r;
      }

      struct Hit {
        std::string path;
        long long mtime = 0;
      };
      std::vector<Hit> hits;
      hits.reserve(64);

      constexpr size_t kLimit = 100;
      bool truncated = false;
      std::unordered_set<std::string> skip_dirs = {".git", "node_modules", "dist", "build", "target", ".venv", "venv"};

      std::error_code ec;
      std::filesystem::recursive_directory_iterator it(
          std::filesystem::path(norm_base), std::filesystem::directory_options::skip_permission_denied, ec);
      std::filesystem::recursive_directory_iterator end;

      for (; it != end && !ec; it.increment(ec)) {
        if (ec) break;
        const auto& p = it->path();
        const bool is_dir = it->is_directory(ec) && !ec;
        if (is_dir) {
          auto name = p.filename().generic_string();
          if (skip_dirs.find(name) != skip_dirs.end()) it.disable_recursion_pending();
          continue;
        }
        const bool is_file = it->is_regular_file(ec) && !ec;
        if (!is_file) continue;

        std::error_code rec_ec;
        auto rel = std::filesystem::relative(p, std::filesystem::path(norm_base), rec_ec).generic_string();
        if (rec_ec) rel = p.filename().generic_string();
        if (!MatchAnyGlob(globs, rel)) continue;

        long long mtime = 0;
        std::error_code time_ec;
        auto ft = std::filesystem::last_write_time(p, time_ec);
        if (!time_ec) mtime = ft.time_since_epoch().count();

        hits.push_back({p.generic_string(), mtime});
        if (hits.size() >= kLimit) {
          truncated = true;
          break;
        }
      }

      std::sort(hits.begin(), hits.end(), [](const Hit& a, const Hit& b) { return a.mtime > b.mtime; });

      std::ostringstream oss;
      if (hits.empty()) {
        oss << "No files found";
      } else {
        for (size_t i = 0; i < hits.size(); i++) {
          oss << hits[i].path;
          if (i + 1 < hits.size()) oss << "\n";
        }
        if (truncated) oss << "\n\n(Results are truncated. Consider using a more specific path or pattern.)";
      }

      r.result = {{"ok", true}, {"title", norm_base}, {"output", oss.str()}, {"metadata", {{"count", hits.size()}, {"truncated", truncated}}}};
      return r;
    });
  }

  {
    ToolSchema schema;
    schema.name = "grep";
    schema.description = "Search file contents using a regex pattern.";
    schema.parameters = {{"type", "object"},
                         {"properties",
                          {{"pattern", {{"type", "string"}}}, {"path", {{"type", "string"}}}, {"include", {{"type", "string"}}}}},
                         {"required", {"pattern"}}};
    reg.RegisterTool(schema, [workspace_root, schema](const std::string& tool_call_id, const nlohmann::json& arguments) {
      ToolResult r;
      r.tool_call_id = tool_call_id;
      r.name = schema.name;
      if (!arguments.is_object() || !arguments.contains("pattern") || !arguments["pattern"].is_string()) {
        r.ok = false;
        r.error = "missing required field: pattern";
        r.result = ErrorResult(r.error);
        return r;
      }

      std::string base = ".";
      if (arguments.contains("path") && arguments["path"].is_string()) base = arguments["path"].get<std::string>();

      std::string norm_base, err;
      if (!NormalizeUnderRoot(workspace_root, base, &norm_base, &err)) {
        r.ok = false;
        r.error = err.empty() ? "invalid path" : err;
        r.result = ErrorResult(r.error);
        return r;
      }

      std::regex pattern;
      try {
        pattern = std::regex(arguments["pattern"].get<std::string>(), std::regex::ECMAScript);
      } catch (const std::exception& e) {
        r.ok = false;
        r.error = std::string("invalid regex: ") + e.what();
        r.result = ErrorResult(r.error);
        return r;
      }

      std::vector<std::regex> include_globs;
      if (arguments.contains("include") && arguments["include"].is_string()) {
        const std::string inc = arguments["include"].get<std::string>();
        try {
          for (const auto& p : ExpandBraceGlob(inc)) include_globs.emplace_back(GlobToRegex(p), std::regex::ECMAScript);
        } catch (...) {
          include_globs.clear();
        }
      }

      struct Match {
        std::string path;
        long long mtime = 0;
        int line = 0;
        std::string text;
      };
      std::vector<Match> matches;
      matches.reserve(64);
      constexpr size_t kLimit = 100;
      constexpr size_t kMaxLineLength = 2000;
      std::unordered_set<std::string> skip_dirs = {".git", "node_modules", "dist", "build", "target", ".venv", "venv"};

      std::error_code ec;
      std::filesystem::recursive_directory_iterator it(
          std::filesystem::path(norm_base), std::filesystem::directory_options::skip_permission_denied, ec);
      std::filesystem::recursive_directory_iterator end;

      bool stop = false;
      for (; it != end && !ec && !stop; it.increment(ec)) {
        if (ec) break;
        const auto& p = it->path();
        const bool is_dir = it->is_directory(ec) && !ec;
        if (is_dir) {
          auto name = p.filename().generic_string();
          if (skip_dirs.find(name) != skip_dirs.end()) it.disable_recursion_pending();
          continue;
        }
        const bool is_file = it->is_regular_file(ec) && !ec;
        if (!is_file) continue;

        std::error_code rec_ec;
        auto rel = std::filesystem::relative(p, std::filesystem::path(norm_base), rec_ec).generic_string();
        if (rec_ec) rel = p.filename().generic_string();
        if (!MatchAnyGlob(include_globs, rel)) continue;

        std::ifstream in(p, std::ios::binary);
        if (!in) continue;

        long long mtime = 0;
        std::error_code time_ec;
        auto ft = std::filesystem::last_write_time(p, time_ec);
        if (!time_ec) mtime = ft.time_since_epoch().count();

        std::string line;
        int line_num = 0;
        while (std::getline(in, line)) {
          line_num++;
          if (!std::regex_search(line, pattern)) continue;
          std::string shown = line;
          if (shown.size() > kMaxLineLength) {
            shown.resize(kMaxLineLength);
            shown += "...";
          }
          matches.push_back({p.generic_string(), mtime, line_num, std::move(shown)});
          if (matches.size() >= kLimit) {
            stop = true;
            break;
          }
        }
      }

      std::sort(matches.begin(), matches.end(), [](const Match& a, const Match& b) { return a.mtime > b.mtime; });

      std::ostringstream oss;
      if (matches.empty()) {
        oss << "No files found";
      } else {
        oss << "Found " << matches.size() << " matches\n";
        std::string current;
        for (size_t i = 0; i < matches.size(); i++) {
          const auto& m = matches[i];
          if (m.path != current) {
            if (!current.empty()) oss << "\n";
            current = m.path;
            oss << current << ":\n";
          }
          oss << "  Line " << m.line << ": " << m.text;
          if (i + 1 < matches.size()) oss << "\n";
        }
      }

      const bool truncated = matches.size() >= kLimit;
      if (truncated) oss << "\n\n(Results are truncated. Consider using a more specific path or pattern.)";

      r.result = {{"ok", true},
                  {"title", arguments["pattern"].get<std::string>()},
                  {"output", oss.str()},
                  {"metadata", {{"matches", matches.size()}, {"truncated", truncated}}}};
      return r;
    });
  }

  {
    ToolSchema schema;
    schema.name = "list";
    schema.description = "List files under a directory.";
    schema.parameters = {{"type", "object"},
                         {"properties", {{"path", {{"type", "string"}}}, {"ignore", {{"type", "array"}, {"items", {{"type", "string"}}}}}}},
                         {"required", nlohmann::json::array()}};
    reg.RegisterTool(schema, [workspace_root, schema](const std::string& tool_call_id, const nlohmann::json& arguments) {
      ToolResult r;
      r.tool_call_id = tool_call_id;
      r.name = schema.name;

      std::string base = ".";
      if (arguments.is_object() && arguments.contains("path") && arguments["path"].is_string()) base = arguments["path"].get<std::string>();

      std::string norm_base, err;
      if (!NormalizeUnderRoot(workspace_root, base, &norm_base, &err)) {
        r.ok = false;
        r.error = err.empty() ? "invalid path" : err;
        r.result = ErrorResult(r.error);
        return r;
      }

      std::vector<std::regex> ignore_globs;
      std::vector<std::string> default_ignores = {"node_modules/**", "__pycache__/**", ".git/**", "dist/**", "build/**",
                                                  "target/**", "vendor/**", "bin/**", "obj/**", ".idea/**", ".vscode/**",
                                                  ".zig-cache/**", "zig-out/**", ".coverage/**", "coverage/**", "tmp/**",
                                                  "temp/**", ".cache/**", "cache/**", "logs/**", ".venv/**", "venv/**", "env/**"};
      try {
        for (const auto& ig : default_ignores) ignore_globs.emplace_back(GlobToRegex(ig), std::regex::ECMAScript);
        if (arguments.is_object() && arguments.contains("ignore") && arguments["ignore"].is_array()) {
          for (const auto& item : arguments["ignore"]) {
            if (!item.is_string()) continue;
            for (const auto& p : ExpandBraceGlob(item.get<std::string>())) {
              ignore_globs.emplace_back(GlobToRegex(p), std::regex::ECMAScript);
            }
          }
        }
      } catch (...) {
        ignore_globs.clear();
      }

      constexpr size_t kLimit = 100;
      std::vector<std::string> files;
      files.reserve(kLimit);

      std::error_code ec;
      std::filesystem::recursive_directory_iterator it(
          std::filesystem::path(norm_base), std::filesystem::directory_options::skip_permission_denied, ec);
      std::filesystem::recursive_directory_iterator end;
      std::unordered_set<std::string> skip_dirs = {".git", "node_modules", "dist", "build", "target", ".venv", "venv"};

      for (; it != end && !ec; it.increment(ec)) {
        if (ec) break;
        const auto& p = it->path();
        const bool is_dir = it->is_directory(ec) && !ec;
        if (is_dir) {
          auto name = p.filename().generic_string();
          if (skip_dirs.find(name) != skip_dirs.end()) it.disable_recursion_pending();
          continue;
        }
        const bool is_file = it->is_regular_file(ec) && !ec;
        if (!is_file) continue;

        std::error_code rec_ec;
        auto rel = std::filesystem::relative(p, std::filesystem::path(norm_base), rec_ec).generic_string();
        if (rec_ec) rel = p.filename().generic_string();
        if (!ignore_globs.empty() && MatchAnyGlob(ignore_globs, rel)) continue;

        files.push_back(rel);
        if (files.size() >= kLimit) break;
      }

      std::sort(files.begin(), files.end());

      std::unordered_set<std::string> dirs;
      dirs.insert(".");
      std::unordered_map<std::string, std::vector<std::string>> files_by_dir;
      for (const auto& f : files) {
        auto pos = f.find_last_of('/');
        std::string dir = (pos == std::string::npos) ? "." : f.substr(0, pos);
        std::string filename = (pos == std::string::npos) ? f : f.substr(pos + 1);
        files_by_dir[dir].push_back(filename);

        if (dir != ".") {
          size_t start = 0;
          while (true) {
            auto slash = dir.find('/', start);
            if (slash == std::string::npos) {
              dirs.insert(dir);
              break;
            }
            dirs.insert(dir.substr(0, slash));
            start = slash + 1;
          }
          dirs.insert(".");
        }
      }

      for (auto& [_, v] : files_by_dir) std::sort(v.begin(), v.end());

      auto render_dir = [&](auto&& self, const std::string& dir_path, int depth) -> std::string {
        std::string out;
        if (depth > 0) {
          out.append(std::string(static_cast<size_t>(depth) * 2, ' '));
          auto pos = dir_path.find_last_of('/');
          out.append((pos == std::string::npos) ? dir_path : dir_path.substr(pos + 1));
          out.append("/\n");
        }

        std::vector<std::string> children;
        children.reserve(dirs.size());
        for (const auto& d : dirs) {
          if (d == "." || d == dir_path) continue;
          auto parent_pos = d.find_last_of('/');
          std::string parent = (parent_pos == std::string::npos) ? "." : d.substr(0, parent_pos);
          if (parent == dir_path) children.push_back(d);
        }
        std::sort(children.begin(), children.end());

        for (const auto& child : children) {
          out += self(self, child, depth + 1);
        }

        auto itf = files_by_dir.find(dir_path);
        if (itf != files_by_dir.end()) {
          for (const auto& fn : itf->second) {
            out.append(std::string(static_cast<size_t>(depth + 1) * 2, ' '));
            out.append(fn);
            out.append("\n");
          }
        }
        return out;
      };

      std::string output;
      output += norm_base;
      if (output.empty() || output.back() != '/') output += "/";
      output += "\n";
      output += render_dir(render_dir, ".", 0);

      r.result = {{"ok", true},
                  {"title", norm_base},
                  {"output", output},
                  {"metadata", {{"count", files.size()}, {"truncated", files.size() >= kLimit}}}};
      return r;
    });
  }

  {
    ToolSchema schema;
    schema.name = "webfetch";
    schema.description = "UNSUPPORTED in local-ai-runtime: fetch web content.";
    schema.parameters = {{"type", "object"},
                         {"properties", {{"url", {{"type", "string"}}}}},
                         {"required", {"url"}}};
    auto handler = [schema](const std::string& tool_call_id, const nlohmann::json&) {
      ToolResult r;
      r.tool_call_id = tool_call_id;
      r.name = schema.name;
      r.ok = false;
      r.error = "webfetch is unsupported";
      r.result = ErrorResult(r.error);
      return r;
    };
    reg.RegisterTool(schema, handler);

    ToolSchema s2 = schema;
    s2.name = "web_fetch";
    reg.RegisterTool(s2, [handler, s2](const std::string& tool_call_id, const nlohmann::json& arguments) mutable {
      auto r = handler(tool_call_id, arguments);
      r.name = s2.name;
      return r;
    });

    ToolSchema s3 = schema;
    s3.name = "WebFetch";
    reg.RegisterTool(s3, [handler, s3](const std::string& tool_call_id, const nlohmann::json& arguments) mutable {
      auto r = handler(tool_call_id, arguments);
      r.name = s3.name;
      return r;
    });
  }

  {
    ToolSchema schema;
    schema.name = "websearch";
    schema.description = "UNSUPPORTED in local-ai-runtime: web search.";
    schema.parameters = {{"type", "object"},
                         {"properties", {{"query", {{"type", "string"}}}, {"num", {{"type", "integer"}}}, {"lr", {{"type", "string"}}}}},
                         {"required", {"query"}}};
    reg.RegisterTool(schema, [schema](const std::string& tool_call_id, const nlohmann::json&) {
      ToolResult r;
      r.tool_call_id = tool_call_id;
      r.name = schema.name;
      r.ok = false;
      r.error = "websearch is unsupported";
      r.result = ErrorResult(r.error);
      return r;
    });
  }

  {
    ToolSchema schema;
    schema.name = "codesearch";
    schema.description = "UNSUPPORTED in local-ai-runtime: code search.";
    schema.parameters = {{"type", "object"},
                         {"properties", {{"query", {{"type", "string"}}}, {"tokensNum", {{"type", "integer"}}}}},
                         {"required", {"query"}}};
    reg.RegisterTool(schema, [schema](const std::string& tool_call_id, const nlohmann::json&) {
      ToolResult r;
      r.tool_call_id = tool_call_id;
      r.name = schema.name;
      r.ok = false;
      r.error = "codesearch is unsupported";
      r.result = ErrorResult(r.error);
      return r;
    });
  }

  {
    ToolSchema schema;
    schema.name = "skill";
    schema.description = "UNSUPPORTED in local-ai-runtime: load skills.";
    schema.parameters = {{"type", "object"},
                         {"properties", {{"name", {{"type", "string"}}}}},
                         {"required", {"name"}}};
    reg.RegisterTool(schema, [schema](const std::string& tool_call_id, const nlohmann::json&) {
      ToolResult r;
      r.tool_call_id = tool_call_id;
      r.name = schema.name;
      r.ok = false;
      r.error = "skill is unsupported";
      r.result = ErrorResult(r.error);
      return r;
    });
  }

  {
    ToolSchema schema;
    schema.name = "question";
    schema.description = "UNSUPPORTED in local-ai-runtime: ask user questions.";
    schema.parameters = {{"type", "object"},
                         {"properties", {{"questions", {{"type", "array"}, {"items", {{"type", "object"}}}}}}},
                         {"required", {"questions"}}};
    reg.RegisterTool(schema, [schema](const std::string& tool_call_id, const nlohmann::json&) {
      ToolResult r;
      r.tool_call_id = tool_call_id;
      r.name = schema.name;
      r.ok = false;
      r.error = "question is unsupported";
      r.result = ErrorResult(r.error);
      return r;
    });
  }

  {
    ToolSchema schema;
    schema.name = "bash";
    schema.description = "UNSUPPORTED in local-ai-runtime: execute shell commands.";
    schema.parameters = {{"type", "object"},
                         {"properties",
                          {{"command", {{"type", "string"}}}, {"timeout", {{"type", "integer"}}}, {"workdir", {{"type", "string"}}}}},
                         {"required", {"command"}}};
    reg.RegisterTool(schema, [schema](const std::string& tool_call_id, const nlohmann::json&) {
      ToolResult r;
      r.tool_call_id = tool_call_id;
      r.name = schema.name;
      r.ok = false;
      r.error = "bash is unsupported";
      r.result = ErrorResult(r.error);
      return r;
    });
  }

  {
    ToolSchema schema;
    schema.name = "terminal";
    schema.description = "UNSUPPORTED in local-ai-runtime: interact with terminal.";
    schema.parameters = {{"type", "object"},
                         {"properties", {{"command", {{"type", "string"}}}}},
                         {"required", {"command"}}};
    reg.RegisterTool(schema, [schema](const std::string& tool_call_id, const nlohmann::json&) {
      ToolResult r;
      r.tool_call_id = tool_call_id;
      r.name = schema.name;
      r.ok = false;
      r.error = "terminal is unsupported";
      r.result = ErrorResult(r.error);
      return r;
    });
  }

  {
    ToolSchema schema;
    schema.name = "task";
    schema.description = "UNSUPPORTED in local-ai-runtime: run a sub-agent task.";
    schema.parameters = {{"type", "object"},
                         {"properties",
                          {{"description", {{"type", "string"}}},
                           {"prompt", {{"type", "string"}}},
                           {"subagent_type", {{"type", "string"}}},
                           {"session_id", {{"type", "string"}}},
                           {"command", {{"type", "string"}}}}},
                         {"required", {"description", "prompt", "subagent_type"}}};
    reg.RegisterTool(schema, [schema](const std::string& tool_call_id, const nlohmann::json&) {
      ToolResult r;
      r.tool_call_id = tool_call_id;
      r.name = schema.name;
      r.ok = false;
      r.error = "task is unsupported";
      r.result = ErrorResult(r.error);
      return r;
    });
  }

  {
    ToolSchema schema;
    schema.name = "todoread";
    schema.description = "UNSUPPORTED in local-ai-runtime: read todo list.";
    schema.parameters = {{"type", "object"}, {"properties", nlohmann::json::object()}, {"required", nlohmann::json::array()}};
    reg.RegisterTool(schema, [schema](const std::string& tool_call_id, const nlohmann::json&) {
      ToolResult r;
      r.tool_call_id = tool_call_id;
      r.name = schema.name;
      r.ok = false;
      r.error = "todoread is unsupported";
      r.result = ErrorResult(r.error);
      return r;
    });
  }

  {
    ToolSchema schema;
    schema.name = "lsp";
    schema.description = "UNSUPPORTED in local-ai-runtime: LSP operations.";
    schema.parameters = {{"type", "object"},
                         {"properties",
                          {{"operation", {{"type", "string"}}},
                           {"filePath", {{"type", "string"}}},
                           {"line", {{"type", "integer"}}},
                           {"character", {{"type", "integer"}}}}},
                         {"required", {"operation", "filePath", "line", "character"}}};
    reg.RegisterTool(schema, [schema](const std::string& tool_call_id, const nlohmann::json&) {
      ToolResult r;
      r.tool_call_id = tool_call_id;
      r.name = schema.name;
      r.ok = false;
      r.error = "lsp is unsupported (use ide.hover/ide.definition/ide.diagnostics if available)";
      r.result = ErrorResult(r.error);
      return r;
    });
  }

  {
    ToolSchema schema;
    schema.name = "batch";
    schema.description = "UNSUPPORTED in local-ai-runtime: batch tool calls.";
    schema.parameters = {{"type", "object"},
                         {"properties",
                          {{"tool_calls",
                            {{"type", "array"},
                             {"items", {{"type", "object"}, {"properties", {{"tool", {{"type", "string"}}}, {"parameters", {{"type", "object"}}}}}}}}}}},
                         {"required", {"tool_calls"}}};
    reg.RegisterTool(schema, [schema](const std::string& tool_call_id, const nlohmann::json&) {
      ToolResult r;
      r.tool_call_id = tool_call_id;
      r.name = schema.name;
      r.ok = false;
      r.error = "batch is unsupported";
      r.result = ErrorResult(r.error);
      return r;
    });
  }

  {
    ToolSchema schema;
    schema.name = "patch";
    schema.description = "UNSUPPORTED in local-ai-runtime: apply a multi-file patch.";
    schema.parameters = {{"type", "object"},
                         {"properties", {{"patchText", {{"type", "string"}}}}},
                         {"required", {"patchText"}}};
    reg.RegisterTool(schema, [schema](const std::string& tool_call_id, const nlohmann::json&) {
      ToolResult r;
      r.tool_call_id = tool_call_id;
      r.name = schema.name;
      r.ok = false;
      r.error = "patch is unsupported";
      r.result = ErrorResult(r.error);
      return r;
    });
  }

  {
    ToolSchema schema;
    schema.name = "multiedit";
    schema.description = "UNSUPPORTED in local-ai-runtime: apply multiple edits to a file.";
    schema.parameters = {{"type", "object"},
                         {"properties",
                          {{"filePath", {{"type", "string"}}},
                           {"edits",
                            {{"type", "array"},
                             {"items",
                              {{"type", "object"},
                               {"properties",
                                {{"filePath", {{"type", "string"}}},
                                 {"oldString", {{"type", "string"}}},
                                 {"newString", {{"type", "string"}}},
                                 {"replaceAll", {{"type", "boolean"}}}}},
                               {"required", {"oldString", "newString"}}}}}}}},
                         {"required", {"filePath", "edits"}}};
    reg.RegisterTool(schema, [schema](const std::string& tool_call_id, const nlohmann::json&) {
      ToolResult r;
      r.tool_call_id = tool_call_id;
      r.name = schema.name;
      r.ok = false;
      r.error = "multiedit is unsupported";
      r.result = ErrorResult(r.error);
      return r;
    });
  }

  {
    ToolSchema schema;
    schema.name = "invalid";
    schema.description = "Invalid tool placeholder.";
    schema.parameters = {{"type", "object"},
                         {"properties", {{"tool", {{"type", "string"}}}, {"error", {{"type", "string"}}}}},
                         {"required", {"tool", "error"}}};
    reg.RegisterTool(schema, [schema](const std::string& tool_call_id, const nlohmann::json& arguments) {
      ToolResult r;
      r.tool_call_id = tool_call_id;
      r.name = schema.name;
      r.ok = false;
      std::string tool = arguments.is_object() && arguments.contains("tool") && arguments["tool"].is_string() ? arguments["tool"].get<std::string>()
                                                                                                            : std::string();
      std::string error = arguments.is_object() && arguments.contains("error") && arguments["error"].is_string()
                              ? arguments["error"].get<std::string>()
                              : std::string();
      if (tool.empty()) tool = "<unknown>";
      if (error.empty()) error = "unknown error";
      r.error = "invalid tool call: " + tool + ": " + error;
      r.result = ErrorResult(r.error);
      return r;
    });
  }

  {
    ToolSchema schema;
    schema.name = "runtime.add";
    schema.description = "Add two numbers and return the sum.";
    schema.parameters = {{"type", "object"},
                         {"properties", {{"a", {{"type", "number"}}}, {"b", {{"type", "number"}}}}},
                         {"required", {"a", "b"}}};
    reg.RegisterTool(schema, [](const std::string& tool_call_id, const nlohmann::json& arguments) {
      ToolResult r;
      r.tool_call_id = tool_call_id;
      r.name = "runtime.add";
      if (!arguments.is_object() || !arguments.contains("a") || !arguments.contains("b")) {
        r.ok = false;
        r.error = "missing required fields: a, b";
        r.result = ErrorResult(r.error);
        return r;
      }
      if (!(arguments["a"].is_number() && arguments["b"].is_number())) {
        r.ok = false;
        r.error = "fields a and b must be numbers";
        r.result = ErrorResult(r.error);
        return r;
      }
      double a = arguments["a"].get<double>();
      double b = arguments["b"].get<double>();
      r.result = {{"ok", true}, {"sum", a + b}};
      return r;
    });
  }

  {
    ToolSchema schema;
    schema.name = "runtime.time";
    schema.description = "Get current unix time in seconds.";
    schema.parameters = {{"type", "object"}, {"properties", nlohmann::json::object()}, {"required", nlohmann::json::array()}};
    reg.RegisterTool(schema, [](const std::string& tool_call_id, const nlohmann::json&) {
      ToolResult r;
      r.tool_call_id = tool_call_id;
      r.name = "runtime.time";
      auto now = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
      r.result = {{"ok", true}, {"unix_seconds", now}};
      return r;
    });
  }

  {
    ToolSchema schema;
    schema.name = "todowrite";
    schema.description = "Write or update a todo list.";
    schema.parameters = {{"type", "object"}, {"properties", nlohmann::json::object()}, {"required", nlohmann::json::array()}};
    reg.RegisterTool(schema, [](const std::string& tool_call_id, const nlohmann::json&) {
      ToolResult r;
      r.tool_call_id = tool_call_id;
      r.name = "todowrite";
      r.result = {{"ok", true}};
      return r;
    });
  }

  return reg;
}

std::vector<std::string> ExtractToolNames(const std::vector<ToolSchema>& tools) {
  std::vector<std::string> out;
  out.reserve(tools.size());
  for (const auto& t : tools) out.push_back(t.name);
  return out;
}

std::optional<nlohmann::json> ParseJsonLoose(const std::string& text) {
  auto trimmed = Trim(text);
  if (trimmed.empty()) return std::nullopt;
  if (auto j = nlohmann::json::parse(trimmed, nullptr, false); !j.is_discarded()) return j;
  if (auto obj = ExtractFirstJsonObject(trimmed)) {
    auto j = nlohmann::json::parse(*obj, nullptr, false);
    if (!j.is_discarded()) return j;
  }
  return std::nullopt;
}

std::optional<std::vector<ToolCall>> ParseToolCallsFromAssistantText(const std::string& assistant_text) {
  llama_agent::ToolCallParser parser;
  auto parsed = parser.parse(assistant_text);
  if (!parsed.empty()) {
    std::vector<ToolCall> out;
    out.reserve(parsed.size());
    for (const auto& c : parsed) {
      ToolCall rc;
      rc.id = c.id;
      rc.name = c.functionName;
      if (c.arguments.is_string()) {
        const auto s = c.arguments.get<std::string>();
        if (ParseJsonLoose(s)) {
          rc.arguments_json = s;
        } else {
          rc.arguments_json = nlohmann::json(s).dump();
        }
      } else if (c.arguments.is_null()) {
        rc.arguments_json = "{}";
      } else {
        rc.arguments_json = c.arguments.dump();
      }
      if (rc.arguments_json.empty()) rc.arguments_json = "{}";
      if (!rc.name.empty()) out.push_back(std::move(rc));
    }
    if (!out.empty()) return out;
  }

  auto jopt = ParseJsonLoose(assistant_text);
  if (jopt) {
    if (auto from_json = ExtractToolCallsFromJson(*jopt)) return from_json;
  }
  if (auto tagged = ExtractToolCallsFromTaggedText(assistant_text)) return tagged;
  if (auto cmd = ExtractToolCallsFromCommandText(assistant_text)) return cmd;
  return ExtractToolCallsFromCatCommandText(assistant_text);
}

}  // namespace runtime
