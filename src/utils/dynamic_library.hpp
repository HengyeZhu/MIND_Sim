#pragma once

#include <memory>
#include <string>

namespace mind_sim::utils {

class DynamicLibrary {
  public:
    explicit DynamicLibrary(std::string path);
    ~DynamicLibrary();

    DynamicLibrary(const DynamicLibrary&) = delete;
    DynamicLibrary& operator=(const DynamicLibrary&) = delete;
    DynamicLibrary(DynamicLibrary&& other) noexcept;
    DynamicLibrary& operator=(DynamicLibrary&& other) noexcept;

    [[nodiscard]] void* symbol(const char* name) const;
    [[nodiscard]] bool has_symbol(const char* name) const noexcept;
    [[nodiscard]] const std::string& path() const noexcept;

  private:
    void close() noexcept;

    std::string path_{};
    void* handle_{nullptr};
};

std::shared_ptr<DynamicLibrary> load_dynamic_library(const std::string& path);

}  // namespace mind_sim::utils
