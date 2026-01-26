#include "session_manager.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

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

class SessionStore {
 public:
  virtual ~SessionStore() = default;
  virtual std::optional<Session> Load(const std::string& session_id) = 0;
  virtual void Save(const Session& s) = 0;
};

namespace {
class FileSessionStore : public SessionStore {
 public:
  FileSessionStore(std::string path, std::string store_namespace)
      : path_(std::move(path)), store_namespace_(std::move(store_namespace)) {
    LoadAll();
  }
  std::optional<Session> Load(const std::string& session_id) override {
    auto it = map_.find(MakeKey(session_id));
    if (it == map_.end()) return std::nullopt;
    return it->second;
  }
  void Save(const Session& s) override {
    map_[MakeKey(s.session_id)] = s;
    PersistAll();
  }

 private:
  std::string path_;
  std::string store_namespace_;
  std::unordered_map<std::string, Session> map_;

  std::string MakeKey(const std::string& session_id) const {
    if (store_namespace_.empty()) return session_id;
    return store_namespace_ + ":" + session_id;
  }

  bool KeyMatchesNamespace(const std::string& key) const {
    if (store_namespace_.empty()) return true;
    if (key.size() <= store_namespace_.size()) return false;
    if (key.rfind(store_namespace_, 0) != 0) return false;
    return key[store_namespace_.size()] == ':';
  }

  std::string StripNamespace(const std::string& key) const {
    if (!KeyMatchesNamespace(key)) return key;
    return key.substr(store_namespace_.size() + 1);
  }

  void LoadAll() {
    std::filesystem::path p(path_);
    std::error_code ec;
    if (!std::filesystem::exists(p, ec)) return;
    std::ifstream in(p, std::ios::binary);
    if (!in) return;
    nlohmann::json j;
    try {
      in >> j;
    } catch (...) {
      return;
    }
    if (!j.is_object() || !j.contains("sessions") || !j["sessions"].is_object()) return;
    for (auto it = j["sessions"].begin(); it != j["sessions"].end(); ++it) {
      if (!it.key().empty() && it.value().is_object()) {
        if (!KeyMatchesNamespace(it.key())) continue;
        Session s;
        s.session_id = StripNamespace(it.key());
        const auto& sj = it.value();
        if (sj.contains("session_id") && sj["session_id"].is_string()) {
          auto sid = sj["session_id"].get<std::string>();
          if (!sid.empty()) s.session_id = sid;
        }
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
        map_.emplace(it.key(), std::move(s));
      }
    }
  }

  nlohmann::json Snapshot() const {
    nlohmann::json out;
    out["sessions"] = nlohmann::json::object();
    for (const auto& it : map_) {
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
        if (t.output_text.has_value()) tj["output_text"] = *t.output_text; else tj["output_text"] = nullptr;
        sj["turns"].push_back(std::move(tj));
      }
      out["sessions"][it.first] = std::move(sj);
    }
    return out;
  }

  void PersistAll() {
    std::filesystem::path path(path_);
    std::error_code ec;
    auto dir = path.parent_path();
    if (!dir.empty()) std::filesystem::create_directories(dir, ec);
    auto tmp = path;
    tmp += ".tmp";
    {
      std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
      if (!out) return;
      out << Snapshot().dump();
    }
    std::filesystem::remove(path, ec);
    std::filesystem::rename(tmp, path, ec);
    if (ec) std::filesystem::remove(tmp, ec);
  }
};

class MiniMemoryStore : public SessionStore {
 public:
  MiniMemoryStore(const HttpEndpoint& ep, const std::string& password, int db, std::string store_namespace)
      : ep_(ep), password_(password), db_(db), store_namespace_(std::move(store_namespace)) {}
  std::optional<Session> Load(const std::string& session_id) override {
    auto conn = Connect();
    if (!conn) return std::nullopt;
    if (!AuthAndSelect(*conn)) return std::nullopt;
    auto val = SendGet(*conn, MakeKey(session_id));
    Close(*conn);
    if (!val.has_value()) return std::nullopt;
    nlohmann::json j = nlohmann::json::parse(*val, nullptr, false);
    if (j.is_discarded() || !j.is_object()) return std::nullopt;
    Session s;
    s.session_id = session_id;
    if (j.contains("history") && j["history"].is_array()) {
      for (const auto& m : j["history"]) {
        if (m.is_object() && m.contains("role") && m["role"].is_string() && m.contains("content")) {
          ChatMessage cm;
          cm.role = m["role"].get<std::string>();
          if (m["content"].is_string()) cm.content = m["content"].get<std::string>();
          s.history.push_back(std::move(cm));
        }
      }
    }
    if (j.contains("turns") && j["turns"].is_array()) {
      for (const auto& t : j["turns"]) {
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
        if (t.contains("output_text") && t["output_text"].is_string()) tr.output_text = t["output_text"].get<std::string>();
        s.turns.push_back(std::move(tr));
      }
    }
    return s;
  }

  void Save(const Session& s) override {
    auto conn = Connect();
    if (!conn) return;
    if (!AuthAndSelect(*conn)) {
      Close(*conn);
      return;
    }
    nlohmann::json j;
    j["history"] = nlohmann::json::array();
    for (const auto& m : s.history) j["history"].push_back({{"role", m.role}, {"content", m.content}});
    j["turns"] = nlohmann::json::array();
    for (const auto& t : s.turns) {
      nlohmann::json tj;
      tj["turn_id"] = t.turn_id;
      tj["input_messages"] = nlohmann::json::array();
      for (const auto& im : t.input_messages) tj["input_messages"].push_back({{"role", im.role}, {"content", im.content}});
      if (t.output_text.has_value()) tj["output_text"] = *t.output_text; else tj["output_text"] = nullptr;
      j["turns"].push_back(std::move(tj));
    }
    auto ok = SendSet(*conn, MakeKey(s.session_id), j.dump());
    Close(*conn);
    (void)ok;
  }

 private:
  HttpEndpoint ep_;
  std::string password_;
  int db_;
  std::string store_namespace_;

  struct Conn {
#ifdef _WIN32
    SOCKET fd = INVALID_SOCKET;
#else
    int fd = -1;
#endif
  };

  std::optional<Conn> Connect() {
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
    Conn c;
    c.fd =
#ifdef _WIN32
        socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
#else
        socket(AF_INET, SOCK_STREAM, 0);
#endif
    if (
#ifdef _WIN32
        c.fd == INVALID_SOCKET
#else
        c.fd < 0
#endif
    ) {
      return std::nullopt;
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(ep_.port);
    if (inet_pton(AF_INET, ep_.host.c_str(), &addr.sin_addr) <= 0) {
      Close(c);
      return std::nullopt;
    }
    if (
#ifdef _WIN32
        ::connect(c.fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR
#else
        ::connect(c.fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0
#endif
    ) {
      Close(c);
      return std::nullopt;
    }
    return c;
  }

  void Close(Conn& c) {
#ifdef _WIN32
    if (c.fd != INVALID_SOCKET) closesocket(c.fd);
    WSACleanup();
#else
    if (c.fd >= 0) close(c.fd);
#endif
    c.fd =
#ifdef _WIN32
        INVALID_SOCKET
#else
        -1
#endif
        ;
  }

  bool SendRaw(Conn& c, const std::string& cmd, std::string* out_resp) {
    if (
#ifdef _WIN32
        send(c.fd, cmd.c_str(), static_cast<int>(cmd.size()), 0) == SOCKET_ERROR
#else
        send(c.fd, cmd.c_str(), cmd.size(), 0) < 0
#endif
    ) {
      return false;
    }
    char buf[8192];
#ifdef _WIN32
    int n = recv(c.fd, buf, sizeof(buf) - 1, 0);
#else
    ssize_t n = recv(c.fd, buf, sizeof(buf) - 1, 0);
#endif
    if (n <= 0) return false;
    buf[n] = '\0';
    if (out_resp) *out_resp = std::string(buf, n);
    return true;
  }

  bool AuthAndSelect(Conn& c) {
    if (!password_.empty()) {
      std::string resp;
      if (!SendRaw(c, Resp({"AUTH", password_}), &resp)) return false;
    }
    if (db_ != 0) {
      std::string resp;
      if (!SendRaw(c, Resp({"SELECT", std::to_string(db_)}), &resp)) return false;
    }
    return true;
  }

  std::optional<std::string> SendGet(Conn& c, const std::string& key) {
    std::string resp;
    if (!SendRaw(c, Resp({"GET", key}), &resp)) return std::nullopt;
    if (resp.rfind("$", 0) == 0) {
      auto crlf = resp.find("\r\n");
      if (crlf == std::string::npos) return std::nullopt;
      int len = std::atoi(resp.substr(1, crlf - 1).c_str());
      if (len < 0) return std::nullopt;
      auto start = crlf + 2;
      if (start + len > resp.size()) return std::nullopt;
      return resp.substr(start, len);
    }
    return std::nullopt;
  }

  bool SendSet(Conn& c, const std::string& key, const std::string& value) {
    std::string resp;
    return SendRaw(c, Resp({"SET", key, value}), &resp);
  }

  std::string MakeKey(const std::string& session_id) const {
    std::string key = "session:";
    if (!store_namespace_.empty()) {
      key += store_namespace_;
      key += ":";
    }
    key += session_id;
    return key;
  }

  std::string Resp(const std::vector<std::string>& args) {
    std::string s = "*" + std::to_string(args.size()) + "\r\n";
    for (const auto& a : args) {
      s += "$" + std::to_string(a.size()) + "\r\n" + a + "\r\n";
    }
    return s;
  }
};
}  // namespace

SessionManager::SessionManager(SessionStoreConfig cfg) {
  std::string store_namespace = cfg.store_namespace;
  if (store_namespace.empty() && cfg.type != "memory") {
    store_namespace = NewId("boot");
  }
  if (cfg.type == "file" && !cfg.file_path.empty()) {
    store_ = std::make_unique<FileSessionStore>(cfg.file_path, store_namespace);
  } else if (cfg.type == "minimemory" || cfg.type == "redis") {
    store_ = std::make_unique<MiniMemoryStore>(cfg.endpoint, cfg.password, cfg.db, store_namespace);
  }
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
  Session out;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = sessions_.find(session_id);
    if (it != sessions_.end()) return it->second;
  }
  bool found = false;
  if (store_) {
    auto loaded = store_->Load(session_id);
    if (loaded) {
      out = *loaded;
      found = true;
    }
  }
  if (!found) out.session_id = session_id;
  {
    std::lock_guard<std::mutex> lock(mu_);
    sessions_[session_id] = out;
  }
  return out;
}

void SessionManager::AppendToHistory(const std::string& session_id, const std::vector<ChatMessage>& messages) {
  Session s;
  {
    std::lock_guard<std::mutex> lock(mu_);
    s = sessions_[session_id];
    s.session_id = session_id;
    s.history.insert(s.history.end(), messages.begin(), messages.end());
    sessions_[session_id] = s;
  }
  if (store_) SaveSessionToStore(s);
}

void SessionManager::AppendTurn(const std::string& session_id, TurnRecord turn) {
  Session s;
  {
    std::lock_guard<std::mutex> lock(mu_);
    s = sessions_[session_id];
    s.session_id = session_id;
    s.turns.emplace_back(std::move(turn));
    sessions_[session_id] = s;
  }
  if (store_) SaveSessionToStore(s);
}

Session SessionManager::LoadSessionFromStore(const std::string& session_id, bool* found) {
  if (found) *found = false;
  if (!store_) return {};
  auto s = store_->Load(session_id);
  if (s) {
    if (found) *found = true;
    return *s;
  }
  return {};
}

void SessionManager::SaveSessionToStore(const Session& session) {
  if (!store_) return;
  store_->Save(session);
}

SessionManager::~SessionManager() {}

}  // namespace runtime
