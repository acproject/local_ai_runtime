#include "session_manager.hpp"

#include <chrono>
#include <random>
#include <sstream>

namespace runtime {
namespace {

static std::string Hex(uint64_t v) {
  std::ostringstream oss;
  oss << std::hex << v;
  return oss.str();
}

static uint64_t Rand64() {
  static thread_local std::mt19937_64 rng(std::random_device{}());
  return rng();
}

}  // namespace

std::string NewId(const std::string& prefix) {
  auto now = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
          .count());
  return prefix + "-" + Hex(now) + "-" + Hex(Rand64());
}

std::string SessionManager::EnsureSessionId(const std::string& preferred) {
  if (!preferred.empty()) return preferred;
  return NewId("sess");
}

Session SessionManager::GetOrCreate(const std::string& session_id) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = sessions_.find(session_id);
  if (it != sessions_.end()) return it->second;
  Session s;
  s.session_id = session_id;
  sessions_.emplace(session_id, s);
  return s;
}

void SessionManager::AppendToHistory(const std::string& session_id, const std::vector<ChatMessage>& messages) {
  std::lock_guard<std::mutex> lock(mu_);
  auto& s = sessions_[session_id];
  s.session_id = session_id;
  s.history.insert(s.history.end(), messages.begin(), messages.end());
}

void SessionManager::AppendTurn(const std::string& session_id, TurnRecord turn) {
  std::lock_guard<std::mutex> lock(mu_);
  auto& s = sessions_[session_id];
  s.session_id = session_id;
  s.turns.emplace_back(std::move(turn));
}

}  // namespace runtime

