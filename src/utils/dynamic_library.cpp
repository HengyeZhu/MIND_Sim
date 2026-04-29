#include "utils/dynamic_library.hpp"

#include <dlfcn.h>

#include <stdexcept>
#include <utility>

namespace mind_sim::utils {

DynamicLibrary::DynamicLibrary(std::string path)
    : path_(std::move(path)) {
    handle_ = dlopen(path_.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle_) {
        throw std::runtime_error("failed to load generated module '" + path_ + "': " + dlerror());
    }
}

DynamicLibrary::~DynamicLibrary() {
    close();
}

DynamicLibrary::DynamicLibrary(DynamicLibrary&& other) noexcept
    : path_(std::move(other.path_)), handle_(other.handle_) {
    other.handle_ = nullptr;
}

DynamicLibrary& DynamicLibrary::operator=(DynamicLibrary&& other) noexcept {
    if (this != &other) {
        close();
        path_ = std::move(other.path_);
        handle_ = other.handle_;
        other.handle_ = nullptr;
    }
    return *this;
}

void* DynamicLibrary::symbol(const char* name) const {
    dlerror();
    void* ptr = dlsym(handle_, name);
    const char* error = dlerror();
    if (error || !ptr) {
        throw std::runtime_error("generated module '" + path_ + "' is missing symbol '" +
                                 std::string(name) + "'");
    }
    return ptr;
}

const std::string& DynamicLibrary::path() const noexcept {
    return path_;
}

void DynamicLibrary::close() noexcept {
    if (handle_) {
        dlclose(handle_);
        handle_ = nullptr;
    }
}

std::shared_ptr<DynamicLibrary> load_dynamic_library(const std::string& path) {
    return std::make_shared<DynamicLibrary>(path);
}

}  // namespace mind_sim::utils
