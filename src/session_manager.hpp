#pragma once

#include "config.hpp"

#include <memory>
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

struct SessionStoreConfig {
  std::string type = "memory";
  std::string file_path;
  HttpEndpoint endpoint;
  std::string password;
  int db = 0;
};

class SessionStore;

class SessionManager {
 public:
  explicit SessionManager(SessionStoreConfig cfg = {});
  ~SessionManager();
  std::string EnsureSessionId(const std::string& preferred);
  Session GetOrCreate(const std::string& session_id);
  void AppendToHistory(const std::string& session_id, const std::vector<ChatMessage>& messages);
  void AppendTurn(const std::string& session_id, TurnRecord turn);

 private:
  Session LoadSessionFromStore(const std::string& session_id, bool* found);
  void SaveSessionToStore(const Session& session);
  std::mutex mu_;
  std::unique_ptr<SessionStore> store_;
  std::unordered_map<std::string, Session> sessions_;
};

std::string NewId(const std::string& prefix);

}  // namespace runtime

