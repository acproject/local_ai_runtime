#pragma once

#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace runtime {

struct ChatMessage {
  std::string role;
  std::string content;
};

struct TurnRecord {
  std::string turn_id;
  std::vector<ChatMessage> input_messages;
  std::optional<std::string> output_text;
};

struct Session {
  std::string session_id;
  std::vector<ChatMessage> history;
  std::vector<TurnRecord> turns;
};

class SessionManager {
 public:
  std::string EnsureSessionId(const std::string& preferred);
  Session GetOrCreate(const std::string& session_id);
  void AppendToHistory(const std::string& session_id, const std::vector<ChatMessage>& messages);
  void AppendTurn(const std::string& session_id, TurnRecord turn);

 private:
  std::mutex mu_;
  std::unordered_map<std::string, Session> sessions_;
};

std::string NewId(const std::string& prefix);

}  // namespace runtime

