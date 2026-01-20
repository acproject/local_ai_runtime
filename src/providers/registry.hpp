#pragma once

#include "providers/provider.hpp"

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace runtime {

class ProviderRegistry {
 public:
  explicit ProviderRegistry(std::string default_provider) : default_provider_(std::move(default_provider)) {}

  void Register(std::unique_ptr<IProvider> provider) {
    if (!provider) return;
    providers_[provider->Name()] = std::move(provider);
  }

  IProvider* Get(const std::string& name) const {
    auto it = providers_.find(name);
    if (it == providers_.end()) return nullptr;
    return it->second.get();
  }

  const std::string& DefaultProviderName() const {
    return default_provider_;
  }

  void SetDefaultProviderName(std::string name) {
    default_provider_ = std::move(name);
  }

  std::vector<IProvider*> List() const {
    std::vector<IProvider*> out;
    out.reserve(providers_.size());
    for (const auto& [_, p] : providers_) out.push_back(p.get());
    return out;
  }

  struct ResolvedModel {
    IProvider* provider = nullptr;
    std::string provider_name;
    std::string model;
  };

  std::optional<ResolvedModel> Resolve(const std::string& model_name) const {
    std::string provider_name;
    std::string model = model_name;
    auto pos = model_name.find(':');
    if (pos != std::string::npos) {
      provider_name = model_name.substr(0, pos);
      model = model_name.substr(pos + 1);
    } else {
      provider_name = default_provider_;
    }
    auto* p = Get(provider_name);
    if (!p) return std::nullopt;
    ResolvedModel r;
    r.provider = p;
    r.provider_name = std::move(provider_name);
    r.model = std::move(model);
    return r;
  }

 private:
  std::string default_provider_;
  std::unordered_map<std::string, std::unique_ptr<IProvider>> providers_;
};

}  // namespace runtime

