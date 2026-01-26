#include "session_manager.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
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

SessionManager::SessionManager(std::string store_path) : store_path_(std::move(store_path)) {
  if (!store_path_.empty()) LoadFromFile();
}

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
  nlohmann::json snapshot;
  Session out;
  bool created = false;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = sessions_.find(session_id);
    if (it != sessions_.end()) return it->second;
    out.session_id = session_id;
    sessions_.emplace(session_id, out);
    created = true;
    if (!store_path_.empty()) snapshot = SnapshotLocked();
  }
  if (created) PersistSnapshot(std::move(snapshot));
  return out;
}

void SessionManager::AppendToHistory(const std::string& session_id, const std::vector<ChatMessage>& messages) {
  nlohmann::json snapshot;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto& s = sessions_[session_id];
    s.session_id = session_id;
    s.history.insert(s.history.end(), messages.begin(), messages.end());
    if (!store_path_.empty()) snapshot = SnapshotLocked();
  }
  PersistSnapshot(std::move(snapshot));
}

void SessionManager::AppendTurn(const std::string& session_id, TurnRecord turn) {
  nlohmann::json snapshot;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto& s = sessions_[session_id];
    s.session_id = session_id;
    s.turns.emplace_back(std::move(turn));
    if (!store_path_.empty()) snapshot = SnapshotLocked();
  }
  PersistSnapshot(std::move(snapshot));
}

nlohmann::json SessionManager::SnapshotLocked() const {
  nlohmann::json out;
  out["sessions"] = nlohmann::json::object();
  for (const auto& it : sessions_) {
    const auto& s = it.second;
    nlohmann::json sj;
    sj["session_id"] = s.session_id;
    sj["history"] = nlohmann::json::array();
    for (const auto& m : s.history) {
      sj["history"].push_back({{"role", m.role}, {"content", m.content}});
    }
    sj["turns"] = nlohmann::json::array();
    for (const auto& t : s.turns) {
      nlohmann::json tj;
      tj["turn_id"] = t.turn_id;
      tj["input_messages"] = nlohmann::json::array();
      for (const auto& im : t.input_messages) {
        tj["input_messages"].push_back({{"role", im.role}, {"content", im.content}});
      }
      if (t.output_text.has_value()) {
        tj["output_text"] = *t.output_text;
      } else {
        tj["output_text"] = nullptr;
      }
      sj["turns"].push_back(std::move(tj));
    }
    out["sessions"][it.first] = std::move(sj);
  }
  return out;
}

void SessionManager::PersistSnapshot(nlohmann::json snapshot) const {
  if (store_path_.empty() || snapshot.is_null()) return;
  std::filesystem::path path(store_path_);
  std::error_code ec;
  auto dir = path.parent_path();
  if (!dir.empty()) std::filesystem::create_directories(dir, ec);
  auto tmp = path;
  tmp += ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) return;
    out << snapshot.dump();
  }
  std::filesystem::remove(path, ec);
  std::filesystem::rename(tmp, path, ec);
  if (ec) std::filesystem::remove(tmp, ec);
}

void SessionManager::LoadFromFile() {
  std::filesystem::path path(store_path_);
  std::error_code ec;
  if (!std::filesystem::exists(path, ec)) return;
  std::ifstream in(path, std::ios::binary);
  if (!in) return;
  nlohmann::json j;
  try {
    in >> j;
  } catch (...) {
    return;
  }
  if (!j.is_object() || !j.contains("sessions") || !j["sessions"].is_object()) return;
  std::unordered_map<std::string, Session> loaded;
  for (auto it = j["sessions"].begin(); it != j["sessions"].end(); ++it) {
    if (!it.key().empty() && it.value().is_object()) {
      Session s;
      s.session_id = it.key();
      const auto& sj = it.value();
      if (sj.contains("session_id") && sj["session_id"].is_string()) s.session_id = sj["session_id"].get<std::string>();
      if (sj.contains("history") && sj["history"].is_array()) {
        for (const auto& m : sj["history"]) {
          if (m.is_object() && m.contains("role") && m["role"].is_string() && m.contains("content")) {
            ChatMessage cm;
            cm.role = m["role"].get<std::string>();
            if (m["content"].is_string()) cm.content = m["content"].get<std::string>();
            s.history.push_back(std::move(cm));
          }
        }
      }
      if (sj.contains("turns") && sj["turns"].is_array()) {
        for (const auto& t : sj["turns"]) {
          if (!t.is_object()) continue;
          TurnRecord tr;
          if (t.contains("turn_id") && t["turn_id"].is_string()) tr.turn_id = t["turn_id"].get<std::string>();
          if (t.contains("input_messages") && t["input_messages"].is_array()) {
            for (const auto& im : t["input_messages"]) {
              if (!im.is_object() || !im.contains("role") || !im["role"].is_string()) continue;
              ChatMessage cm;
              cm.role = im["role"].get<std::string>();
              if (im.contains("content") && im["content"].is_string()) cm.content = im["content"].get<std::string>();
              tr.input_messages.push_back(std::move(cm));
            }
          }
          if (t.contains("output_text") && t["output_text"].is_string()) {
            tr.output_text = t["output_text"].get<std::string>();
          }
          s.turns.push_back(std::move(tr));
        }
      }
      loaded.emplace(s.session_id, std::move(s));
    }
  }
  {
    std::lock_guard<std::mutex> lock(mu_);
    sessions_ = std::move(loaded);
  }
}

}  // namespace runtime

